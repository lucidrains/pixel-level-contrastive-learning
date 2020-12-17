from setuptools import setup, find_packages

setup(
  name = 'pixel-level-contrastive-learning',
  packages = find_packages(),
  version = '0.0.8',
  license='MIT',
  description = 'Pixel-Level Contrastive Learning',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/pixel-level-contrastive-learning',
  keywords = ['self-supervised learning', 'artificial intelligence'],
  install_requires=[
      'einops',
      'torch>=1.6',
      'kornia>=0.4.0'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)