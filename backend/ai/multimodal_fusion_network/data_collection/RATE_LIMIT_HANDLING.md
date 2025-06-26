# Rate Limit Handling in GitHub Data Collection

This document explains how our system handles GitHub API rate limits during the data collection process.

## Overview

GitHub API has strict rate limits:

- 5,000 requests per hour for authenticated users (with token)
- 60 requests per hour for unauthenticated users

Our collection system is designed to properly handle these limits to ensure:

1. No data loss when limits are reached
2. Ability to resume collection after limits reset
3. Efficient use of available request quota

## Implementation Details

### Automatic Rate Limit Detection

The system checks rate limits before making significant API requests using the `_check_rate_limit_before_request()` method. When the available requests drop below 10, collection is automatically paused.

### Exception Handling

When rate limits are reached, a `RateLimitExceededException` is raised with detailed information:

- Exact time when rate limits will reset
- Remaining limits
- Recommended wait time

### State Preservation

Before stopping due to rate limits, the system:

1. Saves all collected data to disk
2. Records the current position (repository, commit, chunk)
3. Writes a state file with detailed metadata
4. Preserves the reset time information

### Automatic Resumption

Using the `auto_collect.py` script, the system can:

1. Detect that collection was stopped due to rate limits
2. Calculate the wait time until reset
3. Sleep until that time arrives (with progress updates)
4. Automatically resume collection from the exact point it stopped

## How to Use

### Manual Handling

If using `collect_100k.py` directly, when rate limits are reached:

1. The script will exit with a message showing when limits will reset
2. You can manually run the script again after that time
3. It will automatically resume from where it left off

### Automatic Handling

Use `auto_collect.py` for fully automated collection:

1. Start collection: `python auto_collect.py`
2. The script handles everything, including waiting for rate limits to reset
3. No manual intervention needed

## Best Practices

1. **Use authenticated tokens**: Always use GitHub API tokens to get 5,000 requests/hour
2. **Multiple tokens**: For faster collection, set up multiple tokens in rotation
3. **Batch processing**: Use smaller batch sizes to avoid losing work when limits are reached
4. **Run overnight**: Start collection before leaving work/sleep to utilize reset periods
5. **Monitor logs**: Check collection logs to see progress and rate limit status

## Troubleshooting

If the system fails to handle rate limits properly:

1. Check that your token is valid and has proper permissions
2. Verify the system clock is accurate (for reset time calculations)
3. Check log files for specific error messages
4. Manually reset collection state if needed by deleting the state file
