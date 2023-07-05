# io_scene_mpet-source_tools - PangYa import script for Blender 2.80+
É um fork de https://github.com/retreev/io_scene_mpet

# Disclaimer
Não sei nada de Modelagem 3D, então tudo que fiz foi baseado no fork original, e pesquisas no google.
Sou novato no Blender ou qualquer outro software de modelagem 3D.
Tentei fazer a animação, mas mesmo se baseando no Valve Blander source tools ainda não conseguiu, o skeletal animation do Blender ainda sai errado.
Pretendo ajeitar isso, depois que pesquisar mais sobre o assunto de animação no Blender com o skeletal animation: https://developer.valvesoftware.com/wiki/Skeletal_animation. 

Pretendo fazer também um leitor do .gbin(gráfico binário) do PangYa, para fazer um Course Studio Tools para carregar todos os Puppet(PET) e objetos no Course do PangYa.
Se não for nesse pluging do Blender, criou outro pluging separado só para isso.

## Export
Ainda não fiz o Export, pretendo implementar depois que conseguir fazer animação, Motions, Frames e Face Animations,
para quando for exportar todos os dados para o arquivo de export, sem usar dados do arquivo de import como o mpetmqo tools.

# PangYa File Format
Para começa usei essa documentação: https://github.com/retreev/Documentation/blob/master/pc/file-formats/pet.md.
E fui ajeitando e adicionando novos valores para todas as versão dos arquivos que faltava.

## Status

  * Funcionando com Blender 2.80+, testado na versão 3.4.
  * Carrega todos os arquivos .pet, .apet, .bpet, .mpet de todas as versões do PangYa do PC.
  * Carrega collision box, mas falta alguns ajustes para ficar 100% alinhado com o bone central.
  * Carrega todas as texturas, porém o especular matérial o nome de arquivos de textura !md_filename, ainda não consegui colocar o .jpg do efeito para o shader na textura do blender.
  * Carrega as animações, porém ainda não conseguiu colocar o skeletal animation corretamente no Blender os valores sai errados.
  * Abre mais de um arquivo de uma vez, só selecionar com shift(depende do sistema operacinal), Os .mpet se já tiver carregado um .bpet(default) ele usar o bone do .bpet(default), se não usar o bone do .mpet.

Todo:

  * Animação: Falta pesquisar como utiliza o skeletal animation no Blender.
  * Motions: É um intervalo nos quadros da animação com metódo de conexão e o proxímo motion. Pretendo implentar depois que conseguir fazer a animação.
  * Frames: É usado para scripts de animação. Pretendo implementar depois que conseguir fazer a animção.
  * Face Animation: É o nome do matérial(textura) que substitui para fazer a animação do rosto. Pretendo implementar depois que conseguir fazer a animação.

## Usage
Clone this git repository into:

  * Windows: `%APPDATA%\Blender Foundation\Blender\2.80+(3.4)\scripts\addons_contrib\io_scene_mpet-source_tools`
  * Linux: `$HOME/.blender/2.80+(3.4)/scripts/addons_contrib/io_scene_mpet-source_tools`

Then activate it in User Preferences (look at the 'Testing' add-ons.)

### What if I don't have git?
Hit [download zip](https://github.com/Acrisio-Filho/io_scene_mpet-source_tools/archive/refs/heads/master.zip) and extract it such that you have a folder named `io_scene_mpet-source_tools` in your `addons_contrib` directory (listed above) and that folder contains `__init__.py` (and neighboring files, of course.)