# How to upgrade python in google cloud instances

\# install pyenv to install python on persistent home directory \
$ curl https://pyenv.run | bash \

\# add to path \
$ echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc \
$ echo 'eval "$(pyenv init -)"' >> ~/.bashrc \
$ echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc \

\# updating bashrc \
$ source ~/.bashrc \

\# install python 3.11.5 and make default \
$ pyenv install 3.11.5 \
$ pyenv global 3.11.5 \
