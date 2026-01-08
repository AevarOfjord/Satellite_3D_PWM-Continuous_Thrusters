# Docker Support Guide

This guide explains how to use Docker to run the Satellite Control System in a containerized environment.

## Quick Start

### Build and Run

```bash
# Build the Docker image
docker build -t satellite-control:latest .

# Run a headless simulation
docker run --rm satellite-control:latest

# Run with custom duration
docker run --rm satellite-control:latest python run_simulation.py run --headless --duration 20.0

# Run with data volume mounted
docker run --rm -v $(pwd)/Data:/app/Data satellite-control:latest
```

### Using Docker Compose

```bash
# Build and run
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Run development container
docker-compose run satellite-dev python run_simulation.py run --help
```

## Docker Images

### Production Image

The production image is optimized for size and includes:
- Python 3.11
- All dependencies
- Application code
- Non-root user for security

**Build:**
```bash
docker build -t satellite-control:latest .
```

**Run:**
```bash
docker run --rm satellite-control:latest
```

### Development Image

The development image mounts source code for live editing:

```bash
docker-compose run satellite-dev bash
```

## Volume Mounts

### Data Directory

Mount the data directory to persist simulation outputs:

```bash
docker run --rm -v $(pwd)/Data:/app/Data satellite-control:latest
```

### Custom Configuration

Mount custom configuration files:

```bash
docker run --rm \
  -v $(pwd)/Data:/app/Data \
  -v $(pwd)/config:/app/config \
  satellite-control:latest
```

## Environment Variables

Set environment variables for configuration:

```bash
docker run --rm \
  -e PYTHONUNBUFFERED=1 \
  -e MUJOCO_GL=egl \
  satellite-control:latest
```

### Available Variables

- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `PYTHONDONTWRITEBYTECODE=1` - Don't write .pyc files
- `MUJOCO_GL=egl` - MuJoCo rendering backend (headless)

## Running Different Commands

### Interactive Shell

```bash
docker run --rm -it satellite-control:latest bash
```

### Run Tests

```bash
docker run --rm satellite-control:latest pytest tests/ -v
```

### Run Benchmarks

```bash
docker run --rm satellite-control:latest pytest tests/benchmarks/ --benchmark-only
```

### View Help

```bash
docker run --rm satellite-control:latest python run_simulation.py --help
```

## Docker Compose Services

### Production Service (`satellite-sim`)

Runs the simulation with production settings:

```bash
docker-compose up satellite-sim
```

### Development Service (`satellite-dev`)

Development service with mounted source code:

```bash
docker-compose run satellite-dev bash
```

## Building Custom Images

### With Different Python Version

```dockerfile
FROM python:3.10-slim
# ... rest of Dockerfile
```

### With Additional Dependencies

```dockerfile
RUN pip install --no-cache-dir additional-package
```

## Troubleshooting

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Fix permissions
sudo chown -R $USER:$USER Data/
```

### MuJoCo Rendering

For headless rendering, ensure `MUJOCO_GL=egl` is set:

```bash
docker run --rm -e MUJOCO_GL=egl satellite-control:latest
```

### Memory Issues

If the container runs out of memory:

```bash
docker run --rm --memory="2g" satellite-control:latest
```

### View Container Logs

```bash
# Docker
docker logs satellite-simulation

# Docker Compose
docker-compose logs -f
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Build Docker image
  run: docker build -t satellite-control:latest .

- name: Run tests in Docker
  run: docker run --rm satellite-control:latest pytest tests/ -v
```

## Best Practices

### 1. Use Multi-Stage Builds

The Dockerfile uses multi-stage builds to reduce image size.

### 2. Non-Root User

The container runs as a non-root user for security.

### 3. Volume Mounts

Mount data directories to persist outputs between runs.

### 4. Environment Variables

Use environment variables for configuration instead of hardcoding.

### 5. Clean Up

Remove unused images and containers:

```bash
docker system prune -a
```

## Examples

### Run Simulation with Custom Parameters

```bash
docker run --rm \
  -v $(pwd)/Data:/app/Data \
  satellite-control:latest \
  python run_simulation.py run \
    --headless \
    --duration 30.0 \
    --start-pos 1.0 1.0 0.0 \
    --target-pos 0.0 0.0 0.0
```

### Run Multiple Simulations

```bash
for i in {1..5}; do
  docker run --rm \
    -v $(pwd)/Data:/app/Data \
    satellite-control:latest \
    python run_simulation.py run --headless --duration 10.0
done
```

### Development Workflow

```bash
# Start development container
docker-compose run satellite-dev bash

# Inside container, edit code and run
python run_simulation.py run --headless

# Exit container (changes persist in mounted volume)
exit
```

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [MuJoCo Docker Guide](https://mujoco.readthedocs.io/en/latest/programming.html#docker)
