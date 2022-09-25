# Contributor Guide

Thank you for your interest in improving this project.
This project is open-source under the [MIT license] and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- [Source Code]
- [Issue Tracker]
- [Code of Conduct]
<!-- - [Documentation] -->

[mit license]: https://opensource.org/licenses/MIT
[source code]: https://github.com/infinitron/epic-stream-processor
[documentation]: https://epic-stream-processor.readthedocs.io/
[issue tracker]: https://github.com/infinitron/epic-stream-processor/issues

## How to report a bug

Report bugs on the [Issue Tracker].

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker].

## How to contribute a feature

Create an issue on the [Issue Tracker] describing the proposed changes. This will allow a chance to talk it over and validate your approach. Make a separate branch, please make sure it has a descriptive name that starts with either the fix/ or feature/ prefixes, for instance, `fix/docs` or `feature/new_storage`. Follow the instructions [here](#how-to-submit-changes) to submit the changes.

## How to set up your development environment

You need Python 3.8+ and the following tools:

- [Poetry]
- [Nox]
- [nox-poetry]

Install the package with development requirements:

```console
$ poetry install
```

You can now run an interactive Python session,
or the command-line interface:

```console
$ poetry run python
$ poetry run epic-stream-processor
```

[poetry]: https://python-poetry.org/
[nox]: https://nox.thea.codes/
[nox-poetry]: https://nox-poetry.readthedocs.io/

## How to add new message types
Add service and message definitions to the ```epic_image.proto``` file and run the following command in the ```epic_grpc``` module folder
```bash
$ python -m grpc_tools.protoc -I./ --python_out=. --mypy_out=. --mypy_grpc_out=. --grpc_python_out=. epic_image.proto
```

## How to test the project

Run the full test suite:

```console
$ nox
```

List the available Nox sessions:

```console
$ nox --list-sessions
```

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

```console
$ nox --session=tests
```

Unit tests are located in the _tests_ directory,
and are written using the [pytest] testing framework.

[pytest]: https://pytest.readthedocs.io/

## How to submit changes

Open a [pull request] to submit changes to this project. Make sure you have a descriptive commit message.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains 100% code coverage.
- If your changes add functionality, update the documentation accordingly.
- You have only one commit (if not, squash them into one commit).

Feel free to submit early, though—we can always iterate on this.

To run linting and code formatting checks before committing your change, you can install pre-commit as a Git hook by running the following command:

```console
$ nox --session=pre-commit -- install
```

[pull request]: https://github.com/infinitron/epic-stream-processor/pulls

<!-- github-only -->

[code of conduct]: CODE_OF_CONDUCT.md
