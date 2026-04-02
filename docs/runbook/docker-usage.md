# Docker Usage Guide

## Building the Container

```bash
# Build the image
docker build -t c5-forecasting:latest .

# Or use docker-compose
docker-compose build
```

## Running Workflows

### Health Check

```bash
docker run --rm c5-forecasting:latest health-check
```

### Validation Workflow

```bash
# Validate raw CSV
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  c5-forecasting:latest validate-raw
```

### Annotation and Dataset Build

```bash
# Annotate dataset
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/artifacts:/app/artifacts \
  c5-forecasting:latest annotate-dataset

# Build raw dataset variant
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/artifacts:/app/artifacts \
  c5-forecasting:latest build-dataset --variant raw
```

### Forecasting

```bash
# Run canary forecast
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/artifacts:/app/artifacts \
  c5-forecasting:latest forecast-next-day
```

### Backtesting

```bash
# Single model backtest
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/artifacts:/app/artifacts \
  c5-forecasting:latest backtest --model frequency_baseline --step 2000

# Full ladder
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/artifacts:/app/artifacts \
  c5-forecasting:latest ladder --step 2000
```

### Champion Comparison

```bash
# Run comparison workflow
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/artifacts:/app/artifacts \
  c5-forecasting:latest compare --step 2000

# Promote champion candidate (dry-run)
docker run --rm \
  -v $(pwd)/artifacts:/app/artifacts \
  c5-forecasting:latest promote

# Promote champion candidate (confirmed)
docker run --rm \
  -v $(pwd)/artifacts:/app/artifacts \
  c5-forecasting:latest promote --confirm --approver "PO"

# View current champion
docker run --rm \
  -v $(pwd)/artifacts:/app/artifacts \
  c5-forecasting:latest champion
```

## Using docker-compose

```bash
# Build
docker-compose build

# Run any command
docker-compose run --rm forecasting health-check
docker-compose run --rm forecasting compare --step 2000
docker-compose run --rm forecasting promote --confirm
```

## Volume Mount Details

| Host Path | Container Path | Access | Purpose |
|-----------|----------------|--------|---------|
| `./data` | `/app/data` | Read-write | Raw CSV, processed Parquet datasets |
| `./configs` | `/app/configs` | Read-only | YAML configuration files |
| `./artifacts` | `/app/artifacts` | Read-write | Run outputs, backtests, comparisons |

## Environment Variables

All `C5_*` environment variables are supported:

```bash
docker run --rm \
  -e C5_LOG_LEVEL=DEBUG \
  -e C5_DATASET_VARIANT=curated \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/artifacts:/app/artifacts \
  c5-forecasting:latest forecast-next-day
```

Or use docker-compose with a `.env` file:

```env
C5_LOG_LEVEL=DEBUG
C5_DATASET_VARIANT=raw
```

## Security Notes

- Container runs as non-root user `c5user` (UID 1000)
- Configs volume is mounted read-only to prevent accidental modification
- No network ports exposed (batch-only system)

## Troubleshooting

### Permission Issues

If you encounter permission errors with artifacts:

```bash
# Create artifacts directory with correct ownership
mkdir -p artifacts
chmod 777 artifacts  # Or: chown 1000:1000 artifacts
```

### Volume Mount Syntax (Windows)

On Windows with PowerShell:

```powershell
docker run --rm `
  -v ${PWD}/data:/app/data:ro `
  -v ${PWD}/artifacts:/app/artifacts `
  c5-forecasting:latest compare --step 2000
```

On Windows with CMD:

```cmd
docker run --rm ^
  -v %cd%/data:/app/data:ro ^
  -v %cd%/artifacts:/app/artifacts ^
  c5-forecasting:latest compare --step 2000
```
