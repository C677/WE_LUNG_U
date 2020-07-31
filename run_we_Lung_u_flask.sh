# change path to start.sh
cd $(cd "$(dirname "$0")" && pwd)

. venv/bin/activate
python start_we_Lung_u.py