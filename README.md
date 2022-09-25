# Epic Stream Processor

[![License](https://img.shields.io/badge/license-MIT-blue)][license]
[![Tests](https://github.com/infinitron/epic-stream-processor/workflows/Tests/badge.svg)][tests]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

<!--[![Codecov](https://codecov.io/gh/infinitron/epic-stream-processor/branch/main/graph/badge.svg)][codecov]-->
<!--[![Read the documentation at https://epic-stream-processor.readthedocs.io/](https://img.shields.io/readthedocs/epic-stream-processor/latest.svg?label=Read%20the%20Docs)][read the docs]-->
<!--[![PyPI](https://img.shields.io/pypi/v/epic-stream-processor.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/epic-stream-processor.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/epic-stream-processor)][python version]-->

[pypi_]: https://pypi.org/project/epic-stream-processor/
[status]: https://pypi.org/project/epic-stream-processor/
[python version]: https://pypi.org/project/epic-stream-processor
[read the docs]: https://epic-stream-processor.readthedocs.io/
[tests]: https://github.com/infinitron/epic-stream-processor/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/infinitron/epic-stream-processor
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

- TODO

## Requirements

- TODO

## Installation

- TODO
<!--You can install _Epic Stream Processor_ via [pip] from [PyPI]:

````console
$ pip install epic-stream-processor
```-->

## Usage

<!-- Please see the [Command-line Reference] for details. -->
Creating a server instance
```python
from epic_stream_processor import server

max_workers = 1
server.serve(max_workers = max_workers)
```

Sending data to the server using a client
```python
from epic_stram_processor.client import EpicRPCClient

rpc_client = EpicRPCClient()
rpc_client.send_dummy_data()

...
hdrs = [...] #list of stringified headers
data = np.ndarray([...]) # data
hdrs.append(d.shape)

rpc_client.send_data()
```

## TODO
- Add documentation
- Add tests
- Enable flake8
- Enable actions
- Update wiki


## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_Epic Stream Processor_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/infinitron/epic-stream-processor/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/infinitron/epic-stream-processor/blob/main/LICENSE
[contributor guide]: https://github.com/infinitron/epic-stream-processor/blob/main/CONTRIBUTING.md
[command-line reference]: https://epic-stream-processor.readthedocs.io/en/latest/usage.html
````
