# Wilbur - Charlotte's Web

### :memo: Sobre o projeto 
Este código foi utilizado como parte do trabalho entitulado _"Quantificação em tempo real do olhar de crianças com autismo por meio de aplicação web"_ e é um piloto de aplicação web baseada em técnicas de Visão Computacional capaz de realizar o rastreamento de olhar das crianças autistas durante a terapia comportamental. 
Almeja-se observar e quantificar o comportamento do olhar em crianças com autismo a fim de auxiliar as métricas de avaliação de desempenho tradicionais utilizadas em terapias baseadas em Applied Behavior Analysis (ABA).

**Disclaimer**: Parte deste projeto foi baseado nas vídeo-aulas de Asadullah Dal, disponível [nesta playlist do YouTube](https://youtu.be/-jFobb6ARc4) sob a Licença MIT.


### :computer: Tecnologias utilizadas
- [x] Python
- [x] OpenCV
- [x] Numpy
- [x] Mediapipe
- [X] VidGear


### :octocat: Instruções de Uso

:one: Faça a instalação e/ou upgrade do sistema de gerenciamento de pacotes para a linguagem Python.

```
python -m pip install pip --upgrade
```

:two: Instale as bibliotecas utilizadas

**OpenCV**: biblioteca de código aberto que inclui várias centenas de algoritmos de visão computacional.
```
pip install opencv-python
pip install opencv-contrib-python
```
**Mediapipe**: soluções de machine learning personalizáveis para mídias e streaming.
```
pip install mediapipe
```

**Vidgear**:framework de processamento de vídeo de alta performanc para construir aplicativos complexos de mídia em tempo real
```
# Instale a versão estável mais recente com todas as dependências do Core
pip install -U vidgear[core]

# Ou instale a versão estável mais recente com todas as dependências Core e Asyncio
pip install -U vidgear[asyncio]
```

:three: Faça um clone desse repositório para a sua máquina

:four: Vá até o caminho da pasta de qual versão você deseja executar: algoritmo ou web

:five: Digite o comando para inicializar. Exemplo:

```
python3 main.py
```
