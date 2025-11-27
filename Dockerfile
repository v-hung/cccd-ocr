# ===== 1. Build React frontend =====
FROM node:22 AS client-build
WORKDIR /frontend
COPY client/ .
ENV BUILD_ENV=build
RUN npm install
RUN npm run build

# ===== 2. Setup Core backend =====

FROM python:3.13

RUN apt-get update && apt-get install -y \
	libgl1 \
	libsm6 \
	libxext6 \
	libfontconfig1 \
	libxrender1 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./.env /code

COPY --from=client-build /frontend/dist /code/app/static

EXPOSE 80

CMD ["fastapi", "run", "app/main.py", "--port", "80"]