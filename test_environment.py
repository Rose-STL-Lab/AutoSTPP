import yaml
import subprocess
import re
from loguru import logger


def regex_matches_version(regex_string, version_string):
    regex_pattern = re.compile(regex_string)
    return regex_pattern.match(version_string) is not None


def main():
    with open('environment.yml', 'r') as f:
        env_data = yaml.safe_load(f)

    # Extract dependencies from the loaded file
    dep_list = env_data['dependencies']

    # Convert list of dependencies to a dictionary
    dep_dict = {}
    for dep in dep_list:
        if isinstance(dep, dict):
            # Handle package dependencies in dictionary format
            dep_dict.update(dep)
        elif isinstance(dep, str):
            if '=' in dep:
                # Handle Python and other dependencies in string format
                key, value = dep.split('=', 1)
            else:
                key = dep
                value = r'.*'
            dep_dict[key.strip()] = value.strip()

    # Run the conda list command and capture the output
    result = subprocess.run(['conda', 'list'], stdout=subprocess.PIPE)

    # Split the output by line and skip the first 2 lines (header)
    lines = result.stdout.decode('utf-8').split('\n')[2:]

    # Parse the package names and versions from each line
    installed_packages = {}
    for line in lines:
        if line and not line.startswith('#'):
            temp = line.split()
            name, version = temp[0], temp[1]
            installed_packages[name] = version

    # Check if the installed package versions match the desired versions
    for name, version in dep_dict.items():
        if '::' in name:
            name = name.split('::')[-1]
        installed_version = installed_packages.get(name)
        if regex_matches_version(version, installed_version):
            logger.debug(f"{name} {installed_version} matches {version}")
        else:
            logger.error(f"{name} {installed_version} does not match {version}")
            raise ModuleNotFoundError(f"{name} {installed_version} does not match {version}")

    logger.info(">>> Conda environment passes all tests!")


if __name__ == '__main__':
    main()
