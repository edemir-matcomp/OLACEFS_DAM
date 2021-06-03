# OLACEFS DAM

Este projeto de pesquisa objetiva desenvolver metodologias para prospectar instrumentos que permitam incorporar geotecnologias no processo de seleção de objetos de controle relacionados à área ambiental.

Especificamente, o projeto se baseia em uma parceria da Organização Latino-Americana e do Caribe de Entidades Fiscalizadoras Superiores (OLACEFS) com o laboratório Pattern Recognition and Earth Observation – Patreo/DCC/UFMG.

São propostos dois temas para serem explorados nesse projeto:
  * Monitoramento de barragens de rejeitos de minérios
  * Identificação de desmatamentos na região amazônica por meio de radar

Para acompanhamento dos temas e entendimento do projeto como um todo, foram gravados os seguintes videos de auxilio.:
  * Identificação de barragens de rejeito de minério: https://youtu.be/IvlA7LI3gSk
  * Monitoramento de desmatamento usando imagens SAR: https://youtu.be/5WQ9wsu9YAs


### Prerequisites

What things you need to install the software and how to install them

```
Docker Engine: https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository
Nvidia-Docker (for gpu support)
```

### Installing

```
# Docker installing steps
sudo apt-get install     apt-transport-https     ca-certificates     curl     software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo docker run hello-world

# Nvidia-Docker installing steps
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |   sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo reboot
```

## Running the docker container

```
# With GPU support
docker run -it --gpus all --ipc=host --name=olacef_container -w /home -v /home:/home edemirfaj/patreo_bot:gpu-py3 bash

# Without GPU support
docker run -it --ipc=host --name=olacef_container -w /home -v /home:/home edemirfaj/patreo_bot:gpu-py3 bash
```

### Instructions

Detailed instructions for each of the project deliveries are in their respective folders
              
1. **1_Script_GEE** - [1_Script_GEE](https://github.com/edemir-matcomp/OLACEFS_DAM/tree/master/1_Script_GEE), [COLAB](https://colab.research.google.com/drive/1exOeSfbCkI0fIIj7hMdhyeY2A3qOiiSd)
2. **2_Classification** - [2_Classification](https://github.com/edemir-matcomp/OLACEFS_DAM/tree/master/2_Classification), [COLAB](https://colab.research.google.com/drive/1bEPgqvYJpsCkspix5ivOy-yCkjoUu_vP?usp=sharing)
3. **3_Script_GEE_Desmatamento** - [3_Script_GEE_Desmatamento](https://github.com/edemir-matcomp/OLACEFS_DAM/tree/master/3_Script_GEE_Desmatamento), [COLAB](https://colab.research.google.com/drive/1gUg_rQLjoGIvsHHUu8LwdGkM0gf0uljo?usp=sharing)
4. **4_Change_Detection** - [4_Change_Detection](https://github.com/edemir-matcomp/OLACEFS_DAM/blob/master/4_Change_Detection/README.md), [COLAB](https://colab.research.google.com/drive/1SWAh0ImS5b7HvX9-e9WpjEUuKG8vZArl?usp=sharing)

### DELIVERY SCHEDULE

| Nº  | Entregas  | Mar-Abr 2020 | Mai-Jul 2020 | Ago-Set 2020 | Out-Dez 2020 
| :------------: |:---------------:| :-----:| :---------------:| :---------------:| :---------------: |
| 1 | Scripts para extração de imagens de barragens de minério | X |  |  |  |
| 2 | Scripts de treinamento, avaliação e uso dos modelos treinados para detecção de barragens de minério | X |  |  |  |
| 3 | Scripts para extração de imagens e dados de desmatamento |  | X |  |  |
| 4 | Scripts de treinamento, avaliação e uso dos modelos treinados para detecção de desmatamento |  |  | X |  |
| 5 | Oficina à distância para transferência de conhecimento  (14 horas de carga) |  |  |  | X |
| 6 | Relatório final |  |  |  | X |

## Authors

* **Alice Gomes** - [gmcalice](https://github.com/gmcalice)
* **Edemir Ferreira** - [edemir-matcomp](https://github.com/edemir-matcomp)
* **Ester Fiorillo** - [esterfiorillo](https://github.com/esterfiorillo)
* **Gabriel Machado** - [gabriellm2003](https://github.com/gabriellm2003)
* **Matheus Brito** - [mbfaria](https://github.com/mbfaria)
* **Pedro Fonseca** - [PedroFW](https://github.com/PedroFW)


## Acknowledgments

* Organização Latino-Americana e do Caribe de Entidades Fiscalizadoras Superiores (OLACEFS)


