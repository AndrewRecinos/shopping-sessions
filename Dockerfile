FROM rust:1 AS build
RUN apt-get update && apt-get install -y --no-install-recommends musl-tools && rm -rf /var/lib/apt/lists/* \
 && rustup target add x86_64-unknown-linux-musl
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main(){}' > src/main.rs \
 && cargo build --release --target x86_64-unknown-linux-musl && rm -rf src
COPY src ./src
RUN touch src/main.rs && cargo build --release --target x86_64-unknown-linux-musl \
 && cp target/x86_64-unknown-linux-musl/release/app /app

FROM scratch
COPY --from=build /app /app
USER 1000:1000
ENTRYPOINT ["/app"]
