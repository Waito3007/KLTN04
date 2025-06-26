# Enhanced GitHub Commit Data Collection System

This module provides an enhanced and optimized system for collecting large-scale GitHub commit data for AI training purposes. The system includes parallel processing capabilities, multiple token support, batch processing, and detailed data analysis tools.

## Features

- **Parallel Processing**: Collect data from multiple repositories simultaneously
- **Multi-Token Support**: Use multiple GitHub tokens to increase rate limits
- **Batch Processing**: Process repositories in batches to manage memory and rate limits
- **Resumable Collection**: Automatically resume collection if interrupted
- **Rate Limit Handling**: Automatically pause and resume when GitHub API rate limits are exceeded
- **Deduplication**: Remove duplicate commits during collection
- **Comprehensive Analysis**: Generate detailed statistics and visualizations from collected data
- **Interactive Mode**: User-friendly CLI interface for non-technical users

## Prerequisites

- Python 3.8 or higher
- Required Python packages:
  - requests
  - matplotlib
  - numpy

## Installation

1. Ensure you have the required Python packages installed:

```bash
pip install requests matplotlib numpy
```

2. Make sure you have at least one GitHub Personal Access Token (classic) with public_repo scope.

## Usage

### 1. Automated Collection with Rate Limit Handling (Recommended)

To automatically collect data with rate limit handling (wait and resume):

```bash
python auto_collect.py
```

This script will:

- Start the data collection process
- Detect when GitHub API rate limits are exceeded
- Wait until the rate limit is reset
- Automatically resume collection from where it left off

### 2. Manual Collection

If you prefer to manage the rate limits manually:

```bash
python collect_100k.py collect --token YOUR_GITHUB_TOKEN --output_dir ../data --target 100000
```

If the collection stops due to rate limits, you can resume it by running the same command later.

### 3. Interactive Mode (Recommended for First-Time Users)

The easiest way to start parallel collection is to use the interactive mode:

```bash
python collect_parallel.py interactive
```

This will guide you through the process of setting up repositories, tokens, and collection parameters.

### 4. Setting Up Repository List

Create a text file with one repository per line in the format `owner/repo`. For example:

```
microsoft/vscode
pytorch/pytorch
tensorflow/tensorflow
```

You can also generate a sample repository list:

```bash
python collect_parallel.py create-files --output_dir ../data
```

### 5. Setting Up GitHub Tokens

Create a text file with one token per line. For example:

```
ghp_your_github_token_1
ghp_your_github_token_2
```

Multiple tokens will help you collect data faster by increasing your rate limit.

### 6. Collecting Data with Multiple Options

To collect data with custom options:

```bash
python collect_parallel.py collect --token_file ../data/tokens.txt --repo_file ../data/repositories.txt --output_dir ../data/parallel_commits --max_commits 1000 --max_workers 4 --batch_size 5
```

Parameters:

- `--token_file`: Path to file containing GitHub tokens
- `--tokens`: List of GitHub tokens (alternative to token_file)
- `--repo_file`: Path to file containing repositories
- `--output_dir`: Directory to save collected data
- `--max_commits`: Maximum commits to collect per repository
- `--max_workers`: Number of parallel workers
- `--batch_size`: Number of repositories per batch
- `--no_deduplicate`: Disable deduplication of commits

### 7. Merging Collected Data

After collection, you can merge all chunk files into a single dataset:

```bash
python collect_100k.py merge --input_dir ../data/github_commits --output_file ../data/all_commits_merged.json
```

### 8. Basic Data Analysis

To analyze the collected data:

```bash
python analyze_data.py --data_dir ../data
```

### 7. Detailed Data Analysis with Visualizations

For comprehensive analysis with visualizations and HTML report:

```bash
python analyze_detailed.py --input ../data/all_commits_merged.json --output ../data/analysis_results
```

This will generate:

- JSON statistics
- Visualizations (charts and graphs)
- HTML report with interactive elements

## Performance Considerations

- **Memory Usage**: Each worker process consumes memory. Adjust `max_workers` based on your system's RAM.
- **API Rate Limits**: GitHub limits API requests to 5,000 per hour per token. Using multiple tokens helps bypass this limitation.
- **Collection Speed**: Parallel collection can be up to 10x faster than sequential collection.
- **Batch Size**: Smaller batch sizes (3-5 repositories) are recommended for better error handling and resource management.

## Example: Collecting 100,000 Commits

To collect approximately 100,000 commits:

1. Create a list of 50-100 active repositories (preferably large projects)
2. Set up at least 2-3 GitHub tokens
3. Run:

```bash
python collect_parallel.py collect --token_file ../data/tokens.txt --repo_file ../data/repositories.txt --max_commits 2000 --max_workers 4
```

Expected timeline:

- With 1 token: 8-12 hours
- With 3 tokens: 3-5 hours
- With 5+ tokens: 1-3 hours

## Comparing with Sequential Collection

| Feature           | Sequential Collector   | Parallel Collector |
| ----------------- | ---------------------- | ------------------ |
| Collection Speed  | 1x                     | 3-10x faster       |
| RAM Usage         | Low                    | Moderate           |
| CPU Usage         | Low                    | High               |
| Resume Capability | Basic                  | Advanced           |
| Deduplication     | Manual post-processing | Built-in           |
| Error Handling    | Basic                  | Comprehensive      |
| Multiple Tokens   | No                     | Yes                |
| Progress Tracking | Basic                  | Detailed           |

## Advanced Usage

### Custom Batch Processing

```bash
python collect_parallel.py collect --tokens ghp_token1 ghp_token2 --repo_file ../data/repos.txt --batch_size 3 --max_workers 3
```

### Handling Very Large Repositories

For repositories with hundreds of thousands of commits (like Linux kernel):

```bash
python collect_parallel.py collect --token_file ../data/tokens.txt --repo_file ../data/large_repos.txt --max_commits 500 --max_workers 2 --batch_size 1
```

## Troubleshooting

- **Rate Limit Errors**: Add more tokens or decrease number of workers
- **Memory Issues**: Reduce batch size and number of workers
- **Collection Interruptions**: Simply restart the script - it will continue from where it left off
- **Missing Data**: Check the log file for any errors during collection

## Contributing

Feel free to submit pull requests or suggest improvements to this system.

## Alternative Method: Git Clone Collector

For environments where GitHub API rate limits are a significant constraint, we provide an alternative collector that uses `git clone` instead of the GitHub API. This approach bypasses API rate limits completely.

### Features of Git Clone Collector

- **No API Rate Limits**: Works without GitHub API tokens
- **Complete History**: Access to all commit information
- **Richer Metadata**: More detailed file change information
- **Batch Processing**: Support for processing repositories in batches
- **Shallow Clone Option**: Minimize disk space and collection time
- **Interactive Mode**: User-friendly interface for non-technical users

### Using Git Clone Collector

#### 1. Interactive Mode (Recommended)

Run the collector in interactive mode for a guided setup:

```bash
python run_git_clone.py
```

This will:

- Guide you through the setup process
- Create a default repository list if needed
- Allow you to customize collection parameters
- Provide a progress display during collection

#### 2. Command Line Usage

Create a repository list:

```bash
python run_git_clone.py create-repo-list --output ../data/repo_list.txt
```

Collect commits from repositories:

```bash
python run_git_clone.py collect --repo_list_file ../data/repo_list.txt --output_file ../data/git_clone_commits.json --max_commits_per_repo 1000 --batch --batch_size 5
```

Parameters:

- `--repo_list_file`: Path to file containing repositories
- `--output_file`: Path to save the collected data
- `--temp_dir`: Temporary directory for cloning repositories
- `--max_commits_per_repo`: Maximum commits to collect per repository
- `--full_clone`: Use full clone instead of shallow clone
- `--batch`: Process repositories in batches
- `--batch_size`: Number of repositories per batch
- `--max_repos`: Maximum number of repositories to process

### Performance Considerations

- **Disk Space**: Git clone requires temporary storage for repositories
- **Processing Time**: Cloning large repositories can take time
- **Network Usage**: Large repositories require more bandwidth

### When to Use Git Clone vs. API

| Consideration              | GitHub API                 | Git Clone                 |
| -------------------------- | -------------------------- | ------------------------- |
| Rate Limits                | Subject to API limits      | No limits                 |
| Speed for many small repos | Faster                     | Slower                    |
| Speed for few large repos  | Slower (due to pagination) | Faster                    |
| Setup Complexity           | Requires tokens            | No tokens needed          |
| Disk Space                 | Minimal                    | Requires space for clones |
| Network Usage              | Less                       | More                      |
| Very Large Scale           | Limited by API             | Virtually unlimited       |

Choose Git Clone collector when:

- You need to collect from many repositories without API rate limit concerns
- You need the complete commit history of repositories
- You don't have (or don't want to use) GitHub API tokens

## License

This project is licensed under the MIT License.
