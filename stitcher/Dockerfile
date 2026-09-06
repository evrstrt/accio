# insta360-mediasdk:3.1.1, the container accio runs the stitch in.
#
# The SDK is licensed and downloaded from Insta360's developer portal
# (Linux CameraSDK + MediaSDK bundle). Put libMediaSDK-dev-3.1.1.0-amd64.tar*.xz
# in this directory, then:
#
#   docker build --platform linux/amd64 -t insta360-mediasdk:3.1.1 stitcher
#
# x86_64 only; on Apple Silicon it runs under emulation.

FROM --platform=linux/amd64 ubuntu:22.04

ADD libMediaSDK-dev-*.xz /tmp/sdk/

RUN apt-get update \
 && apt-get install -y --no-install-recommends /tmp/sdk/libMediaSDK-dev-*/*.deb \
 && mv /tmp/sdk/libMediaSDK-dev-*/models /opt/models \
 && rm -rf /tmp/sdk /var/lib/apt/lists/*

ENTRYPOINT ["MediaSDKTest"]
