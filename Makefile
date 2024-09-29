all:
	@echo "make env	- create env file"
	@echo "make up	- Run apps from docker compose"
	@exit 0

env:
	@if [ -f .env.prod ]; then \
		echo ".env.prod file already exists. Skipping creation."; \
	else \
		echo "Creating .env.prod file from .env_example..."; \
		cp .env_example .env.prod; \
		echo ".env.prod file created successfully."; \
	fi

up:
	@echo "Starting docker-compose..."
	@docker compose -f docker-compose.yml up -d --build
	@echo "Docker-compose started successfully."

stop:
	@echo "Stopping docker-compose..."
	@docker compose -f docker-compose.yml stop
	@echo "Docker-compose stopped successfully."