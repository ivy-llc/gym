# lint as: python3
# Copyright 2021 The Ivy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License..
# ==============================================================================
import setuptools

setuptools.setup(
    name='ivy-gym',
    version='1.1.6',
    author='Ivy Team',
    author_email='ivydl.team@gmail.com',
    description='Fully differentiable reinforcement learning environments, written in Ivy.',
    long_description="""# What is Ivy Gym?\n\nIvy Gym opens the door for intersectional research between supervised
    learning (SL), reinforcement learning (RL), and trajectory optimization (TO), by implementing RL environments in a
    fully differentiable manner.\n\n
    Specifically, Ivy gym provides differentiable implementations of the classic control tasks from OpenAI Gym, as well
    as a new Swimmer task, which illustrates the simplicity of creating new tasks using Ivy. The differentiable nature
    of the environments means that the cumulative reward can be directly optimized for in a supervised manner, without
    need for reinforcement learning, which is the de facto approach for optimizing cumulative rewards. Ivy currently
    supports Jax, TensorFlow, PyTorch, MXNet and Numpy. Check out the [docs](https://ivy-dl.org/gym) for more info!""",
    long_description_content_type='text/markdown',
    url='https://ivy-dl.org/gym',
    project_urls={
        'Docs': 'https://ivy-dl.org/gym/',
        'Source': 'https://github.com/ivy-dl/gym',
    },
    packages=setuptools.find_packages(),
    install_requires=['gym', 'GLU', 'pyglet'],
    classifiers=['License :: OSI Approved :: Apache Software License'],
    license='Apache 2.0'
)
