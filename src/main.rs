use std::{
    borrow::Borrow,
    collections::HashSet,
    error::Error,
    fmt::{write, Display},
    marker::PhantomData,
    ops::Add,
};

use burn::{
    backend::{Autodiff, LibTorch},
    config::Config,
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::Dataset,
    },
    module::Module,
    nn::{
        loss::CrossEntropyLossConfig, BiLstm, BiLstmConfig, Embedding, EmbeddingConfig, Linear,
        LinearConfig, Lstm, LstmConfig, LstmState,
    },
    optim::{decay::WeightDecayConfig, AdamConfig},
    prelude::Backend,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::{backend::AutodiffBackend, Data, Int, Tensor, TensorData},
};

use burn_train::{
    metric::{
        store::{Aggregate, Direction, Split},
        AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric,
    },
    ClassificationOutput, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    TrainOutput, TrainStep, ValidStep,
};
use regex::Regex;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use uuid::Uuid;

type Res = Result<(), Box<dyn Error>>;
type ResT<T> = Result<T, Box<dyn Error>>;

#[derive(Serialize, Deserialize, Debug)]
struct Address {
    #[serde(alias = "house_uuid")]
    uuid: Uuid,

    #[serde(alias = "house_full_address")]
    address: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Shutdown {
    #[serde(alias = "shutdown_id")]
    id: u32,

    #[serde(alias = "comment")]
    comment: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ShutdownCleared {
    #[serde(alias = "shutdown_id")]
    id: u32,

    #[serde(default, skip_serializing)]
    comment_words: Vec<String>,

    #[serde(alias = "words_str")]
    comment_words_str: String,

    #[serde(default, skip_serializing)]
    tags: Vec<Tag>,

    #[serde(alias = "tags_str")]
    tags_str: String,
}

impl ShutdownCleared {
    fn prepare_for_serialize(&mut self) {
        self.comment_words_str = self.comment_words.join(",");
        self.tags_str = self
            .tags
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
    }

    fn after_deserialize(&mut self) {
        self.comment_words = self
            .comment_words_str
            .split(",")
            .map(|x| x.to_string())
            .collect::<Vec<_>>();
        self.tags = self
            .tags_str
            .split(",")
            .filter_map(|x| match x {
                "O" => Some(Tag::O),
                "Street" => Some(Tag::Street),
                "Number" => Some(Tag::Number),
                "Range" => Some(Tag::Range),
                "Even" => Some(Tag::Even),
                "Odd" => Some(Tag::Odd),
                _ => None,
            })
            .collect::<Vec<_>>()
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash, Clone, PartialOrd, Ord)]
enum Tag {
    O,
    Street,
    Number,
    Range,
    Even,
    Odd,
}

#[derive(Debug, Serialize)]
struct TokenizedInput {
    #[serde(default, skip_serializing)]
    x: Vec<u64>,
    #[serde(alias = "x")]
    x_str: String,
    #[serde(default, skip_serializing)]
    y: Vec<u64>,
    #[serde(alias = "y")]
    y_str: String
}

impl Display for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

const RANGE_PATTERN: &str = r"(\d+)\s*-\s*(\d+)";
const STREET_PATTERN: &str = r"Ульяновская обл, Ульяновск г, ";
const EVEN_PATTERN: &str = r"(\W|^)(чет|чёт)";
const ODD_PATTERN: &str = r"(\W|^)неч";
const NUMBER_PATTERN: &str = r"\d+/?\d+\w?";

#[derive(Module, Debug)]
struct Model<B: Backend> {
    embedding: Embedding<B>,
    bi_lstm: BiLstm<B>,
    lstm: Lstm<B>,
    dense: Linear<B>,
    bi_lstm_cell: Option<Tensor<B, 3>>,
    bi_lstm_hidden: Option<Tensor<B, 3>>,
    lstm_cell: Option<Tensor<B, 2>>,
    lstm_hidden: Option<Tensor<B, 2>>,
}

impl<B: Backend> Model<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        bi_lstm_state: Option<LstmState<B, 3>>,
        lstm_state: Option<LstmState<B, 2>>,
    ) -> (Tensor<B, 3>, LstmState<B, 3>, LstmState<B, 2>) {
        let x = self.embedding.forward(input);
        let x_bilstm = self.bi_lstm.forward(x, bi_lstm_state);
        let x_lstm = self.lstm.forward(x_bilstm.0, lstm_state);
        let x = self.dense.forward(x_lstm.0);
        (x, x_bilstm.1, x_lstm.1)
    }

    pub fn forward_classification(&self, item: ShutdownBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(
            targets.clone(),
            match (&self.bi_lstm_cell, &self.bi_lstm_hidden) {
                (Some(l), Some(r)) => Some(LstmState::new(l.clone(), r.clone())),
                _ => None,
            },
            match (&self.lstm_cell, &self.lstm_hidden) {
                (Some(l), Some(r)) => Some(LstmState::new(l.clone(), r.clone())),
                _ => None,
            },
        );

        let targets = targets.clone().squeeze(1);
        let output = output.0.clone().squeeze(2);

        let loss = CrossEntropyLossConfig::new()
            .init(&targets.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<ShutdownBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: ShutdownBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ShutdownBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: ShutdownBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            embedding: EmbeddingConfig::new(4544, 64).init(device),
            bi_lstm: BiLstmConfig::new(64, 64, true).init(device),
            lstm: LstmConfig::new(64, 256, true).init(device),
            dense: LinearConfig::new(6, 256).init(device),
            bi_lstm_cell: None,
            bi_lstm_hidden: None,
            lstm_cell: None,
            lstm_hidden: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ShutdownBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct ShutdownBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> ShutdownBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ShutdownItem {
    inputs: Vec<u64>,
    targets: Vec<u64>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ShutdownDataset {
    dataset: Vec<ShutdownItem>,
}

impl Dataset<ShutdownItem> for ShutdownDataset {
    fn get(&self, index: usize) -> Option<ShutdownItem> {
        self.dataset.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl<B: Backend> Batcher<ShutdownItem, ShutdownBatch<B>> for ShutdownBatcher<B> {
    fn batch(&self, input: Vec<ShutdownItem>) -> ShutdownBatch<B> {
        let input_data = input
            .iter()
            .flat_map(|x| x.inputs.clone())
            .collect::<Vec<_>>();
        let input_t = TensorData::new(
            input_data,
            [input.len(), input.iter().next().unwrap().inputs.len()],
        );

        let target_data = input
            .iter()
            .flat_map(|x| x.targets.clone())
            .collect::<Vec<_>>();
        let target_t = TensorData::new(
            target_data,
            [input.len(), input.iter().next().unwrap().targets.len()],
        );

        ShutdownBatch {
            inputs: Tensor::from_data(input_t, &self.device),
            targets: Tensor::from_data(target_t, &self.device),
        }
    }
}

fn main() -> Res {
    let addresses_path = std::env::var("ADDRESES")?;
    let shutdown_path = &std::env::var("SHUTDOWN")?;

    let mut addresses = get_from_csv::<Address>(&addresses_path)?;
    let shutdowns = get_from_csv::<Shutdown>(&shutdown_path)?;

    let stop_words_regex = Regex::new(&get_shutdown_stopwords_pattern())?;

    let split_symbols = [' ', ',', ';'];

    for address in &mut addresses {
        address.address = address.address.replace(STREET_PATTERN, "").to_lowercase();
    }

    let range_regex = Regex::new(RANGE_PATTERN)?;
    let even_regex = Regex::new(EVEN_PATTERN)?;
    let odd_regex = Regex::new(ODD_PATTERN)?;
    let number_regex = Regex::new(NUMBER_PATTERN)?;

    println!("Shutdowns count: {}", shutdowns.len());

    let shutdowns_cleared;

    if let Ok(mut shutdowns_cleared_ok) =
        get_from_csv::<ShutdownCleared>("tagged.csv").or(get_from_csv("../../tagged.csv"))
    {
        for shutdown in &mut shutdowns_cleared_ok {
            shutdown.after_deserialize();
        }

        shutdowns_cleared = shutdowns_cleared_ok;
    } else {
        shutdowns_cleared = shutdowns
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                let comment_words =
                    to_words(&x.comment, &stop_words_regex, &range_regex, &split_symbols);

                if i % 100 == 0 {
                    println!("Finished: {}", i);
                }

                let tags = comment_words
                    .iter()
                    .map(|x| {
                        if even_regex.is_match(x) {
                            Tag::Even
                        } else if odd_regex.is_match(x) {
                            Tag::Odd
                        } else if range_regex.is_match(x) {
                            Tag::Range
                        } else if number_regex.is_match(x) {
                            Tag::Number
                        } else if addresses.iter().any(|y| y.address.contains(x)) {
                            Tag::Street
                        } else {
                            Tag::O
                        }
                    })
                    .collect::<Vec<_>>();

                let mut result = ShutdownCleared {
                    id: x.id,
                    comment_words,
                    comment_words_str: String::default(),
                    tags,
                    tags_str: String::default(),
                };

                result.prepare_for_serialize();

                result
            })
            .collect::<Vec<_>>();

        let mut writer = csv::WriterBuilder::new()
            .has_headers(true)
            .delimiter(b';')
            .from_path("tagged.csv")?;

        for shutdown in &shutdowns_cleared {
            writer.serialize(shutdown)?;
        }
    }

    let mut words_dict = shutdowns_cleared
        .iter()
        .map(|x| x.comment_words.iter().map(|y| y.as_str()))
        .flat_map(|x| x)
        .collect::<Vec<_>>();

    words_dict.sort();
    words_dict.dedup();

    let mut tags_dict = shutdowns_cleared
        .iter()
        .map(|x| x.tags.clone())
        .flat_map(|x| x)
        .collect::<Vec<_>>();

    tags_dict.sort();
    tags_dict.dedup();

    let tokenized = shutdowns_cleared
        .iter()
        .map(|shutdown| {
            let mut input = TokenizedInput {
                x: vec![0;256],
                y: vec![0;256],
                x_str: String::default(),
                y_str: String::default()
            };

            for (i, j) in (0..256).rev().enumerate() {
                if i < shutdown.comment_words.len() && i < shutdown.tags.len() {
                    input.x[j] = match words_dict
                        .iter()
                        .position(|y| y == &shutdown.comment_words[i].as_str())
                    {
                        Some(v) => v as u64 + 1,
                        None => 0,
                    };
                    input.y[j] = match tags_dict.iter().position(|y| y == &shutdown.tags[i]) {
                        Some(v) => v as u64 + 1,
                        None => 0,
                    };
                }
            }

            input.x_str = input.x.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
            input.y_str = input.y.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");

            input
        })
        .collect::<Vec<_>>();

    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .delimiter(b';')
        .from_path("tagged-we.csv")?;

    for shutdown in &tokenized {
        writer.serialize(shutdown)?;
    }

    let max_input = shutdowns_cleared
        .iter()
        .map(|x| x.comment_words.len())
        .max()
        .unwrap();

    println!(
        "Words in dict: {}\nClasses: {}\nMax input toknes length: {}",
        words_dict.len(),
        tags_dict.len(),
        max_input
    );

    let dataset = tokenized
        .iter()
        .map(|x| ShutdownItem {
            inputs: x.x.clone(),
            targets: x.y.clone(),
        })
        .collect::<Vec<_>>();

    let dataset = ShutdownDataset { dataset };

    println!("Words dict len: {}\nTags dict len: {}", words_dict.len(), tags_dict.len());

    run(,dataset)

    // for token in tokenized {
    //     let x = model.forward(
    //         Tensor::from_data(
    //             TensorData::new(token.x.into_iter().collect::<Vec<u64>>(), &[16, 16]),
    //             device,
    //         ),
    //         bilstm,
    //         lstm,
    //     );
    //     bilstm = Some(x.1);
    //     lstm = Some(x.2);
    // }

    Ok(())
}
pub fn run<B: AutodiffBackend>(device: B::Device, dataset: ShutdownDataset) {
    let device = Default::default();
    let model = ModelConfig::new(6, 64).init::<B>(&device);
    let batcher = ShutdownBatcher::<B>::new(device.clone());
    let batcher_valid = ShutdownBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher)
        .batch_size(64)
        .shuffle(2)
        .num_workers(4)
        .build(dataset.clone());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(64)
        .shuffle(2)
        .num_workers(4)
        .build(dataset);

    let config_optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));

    let learner = LearnerBuilder::new("/")
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<Autodiff<LibTorch>>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device])
        .num_epochs(10)
        .summary()
        .build(model, config_optimizer.init(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_test);
}
fn to_words(
    input: &str,
    stop_words_regex: &Regex,
    range_regex: &Regex,
    split_symbols: &[char],
) -> Vec<String> {
    let regexed = stop_words_regex.replace_all(&input, " ");
    let regexed = range_regex.replace_all(&regexed, "$1-$2");

    regexed
        .to_lowercase()
        .split(split_symbols)
        .filter(|x| !x.is_empty())
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
}

fn get_from_csv<R: DeserializeOwned>(path: &str) -> ResT<Vec<R>> {
    Ok(csv::ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .from_path(path)?
        .deserialize::<R>()
        .filter_map(|x| x.ok())
        .collect::<Vec<_>>())
}

fn get_shutdown_stopwords_pattern() -> String {
    let stop_words = [
        "из-под",
        r"д[-=\s](\d{2,3})?(мм)?;?",
        "хвс",
        "гвс",
        "ж/д",
        "без",
        "задвижки",
        "ремонт",
        "замена",
        "ввода",
        "утечка",
        "из",
        "колодца",
        "земли",
        "метров",
        "трубы(-{2,3})?",
        r"ч[/\\]д",
        r"п[/\\]з",
        "кол-ца",
        "и",
        "в/к;?",
        "в",
        "-й",
        "-ая",
        r"\(\s*д/с(ад)?\s*№?\d{1,3}\s*[и,]\s*№?\d{1,3}\s*\)",
        "д/с",
        "труб",
        "пониж",
        "давл",
        "на",
        "до",
        "цтп",
    ];

    format!(
        r"(^|\b)[\s\.,-]*(?i)({})[\s\.,-]*(\b|$)",
        stop_words.join("|")
    )
}
