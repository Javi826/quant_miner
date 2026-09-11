FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99

RUN apt-get update && apt-get install -y \
    wget curl ca-certificates \
    xvfb x11vnc \
    libx11-6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install
RUN wget -q "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5linux64.sh" \
    && chmod +x mt5linux64.sh \
    && xvfb-run --auto-servernum ./mt5linux64.sh --install \
    && rm mt5linux64.sh

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5900
VOLUME ["/mt5/experts", "/mt5/logs", "/mt5/config"]
ENTRYPOINT ["/entrypoint.sh"]
