use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// The CLI command struct
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
pub struct Cli {
    /// command
    #[clap(subcommand)]
    pub command: Commands,
}

/// factory method for command line args
pub fn get_cli() -> Cli {
    Cli::parse()
}

/// The subcommands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Calibrate the filter on control sequences
    Calibrate {

        /// The control sequence file to read from
        #[clap(short, long, parse(from_os_str))]
        input: PathBuf,

        /// The filter coefficient file
        #[clap(short, long, parse(from_os_str))]
        filter: PathBuf,

        /// The output file name
        #[clap(short, long, default_value="train.npy", parse(from_os_str))]
        output: PathBuf,

        /// The number of iterations to run
        #[clap(long, default_value = "0")]
        iterations: usize,

        /// The number of iterations to run
        #[clap(short, long, default_value = "1")]
        batch_size: usize,

        /// The number of iterations to run
        #[clap(short, long, default_value = "1e-6")]
        step_size: f64,
    },
    /// Validate the filter on control sequences
    Validate {
        /// The control sequence file to read from
        #[clap(short, long, parse(from_os_str))]
        input: PathBuf,

        /// The filter coefficient file
        #[clap(short, long, parse(from_os_str))]
        filter: PathBuf,

        /// The output file name
        #[clap(short, long, default_value="test.npy", parse(from_os_str))]
        output: PathBuf,
    },
}
