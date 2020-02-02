try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
import uuid

from setuptools import find_packages, setup

install_reqs = parse_requirements('requirements.txt', session=uuid.uuid1())
reqs = [str(req.req) for req in install_reqs]

setup(
    name='bumblebeat',
    version="1.0",
    packages=find_packages(),
    author_email='thomasgnuttall@gmail.com',
    zip_safe=False,
    include_package_data=True,
    long_description=open('README.md').read(),
    install_requires=reqs
)
