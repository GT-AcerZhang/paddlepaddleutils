apt-get install -y wget vim git
apt-get cleanup

mkdir /soft/
cd /soft
wget https://storage.googleapis.com/golang/go1.8.linux-amd64.tar.gz
tar -xzvf go1.8.linux-amd64.tar.gz
