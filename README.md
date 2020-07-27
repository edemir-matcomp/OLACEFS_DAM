# OLACEFS DAM

Este projeto de pesquisa objetiva desenvolver metodologias para prospectar instrumentos que permitam incorporar geotecnologias no processo de seleção de objetos de controle relacionados à área ambiental.

Especificamente, o projeto se baseia em uma parceria da Organização Latino-Americana e do Caribe de Entidades Fiscalizadoras Superiores (OLACEFS) com o laboratório Pattern Recognition and Earth Observation – Patreo/DCC/UFMG.

São propostos dois temas para serem explorados nesse projeto:
  * Monitoramento de barragens de rejeitos de minérios
  * Identificação de desmatamentos na região amazônica por meio de radar


### Prerequisites

What things you need to install the software and how to install them

```
Docker Engine
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Docker Engine: https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository
Nvidia-Docker (for gpu support)
```

And repeat

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

End with an example of getting some data out of the system or using it for a little demo

## Running the docker container

```
docker run -it --gpus all --ipc=host --name=olacef_container -w /home -v /home:/home edemirfaj/patreo_bot:gpu-py3 bash
```

### Instructions

Detailed instructions for each of the project deliveries are in their respective folders
              
1. **1_Script_GEE** - [1_Script_GEE](https://github.com/edemir-matcomp/OLACEFS_DAM/tree/master/1_Script_GEE)
2. **2_Classification** - [2_Classification](https://github.com/edemir-matcomp/OLACEFS_DAM/tree/master/2_Classification)
3. **3_Script_GEE_Desmatamento** - [3_Script_GEE_Desmatamento](https://github.com/edemir-matcomp/OLACEFS_DAM/tree/master/3_Script_GEE_Desmatamento)

```
https://github.com/edemir-matcomp/OLACEFS_DAM/tree/master/1_Script_GEE
```

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

* **Alice Gomes** - []()
* **Edemir Ferreira** - [edemir-matcomp](https://github.com/edemir-matcomp)
* **Ester Fiorillo** - [esterfiorillo](https://github.com/esterfiorillo)
* **Matheus Brito** - [mbfaria](https://github.com/mbfaria)
* **Pedro Fonseca** - [PedroFW](https://github.com/PedroFW)


## Acknowledgments

* Organização Latino-Americana e do Caribe de Entidades Fiscalizadoras Superiores (OLACEFS)


