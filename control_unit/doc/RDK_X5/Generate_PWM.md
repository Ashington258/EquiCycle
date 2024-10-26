现象描述：
我的默认python环境中可以import Hobot.GPIO as GPIO但是这是一个不能直接通过pip安装的包，我在/usr/lib/hobot-gpio/lib/python/Hobot/GPIO/路径下找到了该包

包目录结构：
📦hobot-gpio
 ┣ 📂lib
 ┃ ┗ 📂python
 ┃ ┃ ┣ 📂Hobot
 ┃ ┃ ┃ ┣ 📂GPIO
 ┃ ┃ ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┃ ┃ ┣ 📜gpio.py
 ┃ ┃ ┃ ┃ ┣ 📜gpio_event.py
 ┃ ┃ ┃ ┃ ┗ 📜gpio_pin_data.py
 ┃ ┃ ┃ ┗ 📜__init__.py
 ┃ ┃ ┣ 📂Hobot.GPIO.egg-info
 ┃ ┃ ┃ ┣ 📜PKG-INFO
 ┃ ┃ ┃ ┣ 📜SOURCES.txt
 ┃ ┃ ┃ ┣ 📜dependency_links.txt
 ┃ ┃ ┃ ┗ 📜top_level.txt
 ┃ ┃ ┗ 📂RPi
 ┃ ┃ ┃ ┣ 📂GPIO
 ┃ ┃ ┃ ┃ ┗ 📜__init__.py
 ┃ ┃ ┃ ┗ 📜__init__.py
 ┣ 📜LICENCE.txt
 ┣ 📜MANIFEST.in
 ┣ 📜README.md
 ┗ 📜setup.py

我应该如何在我的虚拟环境中(/root/miniforge-pypy3/envs/control/bin/python)导入该包，并且正常驱动pwm


如果需要的化我提供setup.py文件：
```python

from setuptools import setup

classifiers = ['Operating System :: POSIX :: Linux',
               'License :: OSI Approved :: MIT License',
               'Intended Audience :: Developers',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Topic :: Software Development',
               'Topic :: System :: Hardware']

setup(name                          = 'Hobot.GPIO',
      version                       = '0.0.2',
      author                        = 'HORIZON',
      author_email                  = 'technical_support@horizon.ai',
      description                   = 'A module to control Hobot GPIO channels',
      long_description              = open('README.md').read(),
      long_description_content_type = 'text/markdown',
      license                       = 'MIT',
      keywords                      = 'Hobot GPIO',
      url                           = '',
      classifiers                   = classifiers,
      package_dir                   = {'': 'lib/python/'},
      packages                      = ['Hobot', 'Hobot.GPIO', 'RPi', 'RPi.GPIO'],
      package_data                  = {'Hobot.GPIO': []},
      include_package_data          = True,
)
```