import os
import subprocess
import argparse


EXTENSIONS = [
    "ms-python.python",
    "jithurjacob.nbpreviewer"
]


class VSCode:
    def __init__(self, port=10000, password=None):
        self.port = port
        self.password = password
        self._install_code()
        self._install_extensions()
        self._run_code()

    def _install_code(self):
        subprocess.run(
            ["wget", "https://code-server.dev/install.sh"], stdout=subprocess.PIPE
        )
        subprocess.run(["sh", "install.sh"], stdout=subprocess.PIPE)

    def _install_extensions(self):
        for ext in EXTENSIONS:
            subprocess.run(["code-server", "--install-extension", f"{ext}"])

    def _run_code(self):
        if self.password:
            code_cmd = f"PASSWORD={self.password} code-server --port {self.port} --disable-telemetry"
        else:
            code_cmd = f"code-server --port {self.port} --auth none --disable-telemetry"
        with subprocess.Popen(
            [code_cmd],
            shell=True,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        ) as proc:
            for line in proc.stdout:
                print(line, end="")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=10000)
    parser.add_argument('--password', type=str, default=None)
    args = parser.parse_args()

    vs = VSCode(port=args.port, password=args.password)
