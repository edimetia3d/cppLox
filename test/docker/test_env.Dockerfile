FROM dart:stable
RUN apt-get update && apt-get install -y lsb-release wget software-properties-common gpg cmake
RUN wget https://apt.llvm.org/llvm.sh && bash llvm.sh 13
RUN apt-get install -y libc++-13-dev libc++abi-13-dev python3 python3-venv python3-pip
