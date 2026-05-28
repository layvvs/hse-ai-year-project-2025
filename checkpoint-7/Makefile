.PHONY: build run stop logs restart clean health


build:
	docker compose build

run: build
	docker compose up -d
	@echo "Frontend:  http://localhost:8501"
	@echo "Backend:   http://localhost:8000"
	@echo "Swagger:   http://localhost:8000/docs"

stop:
	docker compose down

restart: stop run

logs:
	docker compose logs -f

health:
	@curl -sf http://localhost:8000/health | python3 -m json.tool

clean:
	docker compose down --rmi local -v
