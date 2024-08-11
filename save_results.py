import sys
import os
import re
import enum

# Using enum class create enumerations
class Status(enum.Enum):
   NOVO = 1 # arquivo zerado ou só com título ou uma nova classificação
   CONCLUIR_RODADAS = 2 # arquivo com menos rodadas desejadas
   CALCULAR_PARAMETROS = 3 # Calcular os parâmetros de média final e outros
   FALTA_VALIDAR = 4 # Falta somente a validação do MNIST

class FileResult:

   def __init__(self, directory, name_file, MAX_RODADAS):
      self.directory = directory + '/'

      # Criando pasta para salvar os resultados
      self.create_directory( self.directory )

      # path completo do arquivo para salvar os dados
      self.path = self.directory  + name_file

      # Analisa se o arquivo já foi criado no passado
      self.__analisa_arquivo(self.path, MAX_RODADAS)

   def __analisa_arquivo(self, path, MAX_RODADAS):

      self.status = Status.NOVO

      try:
         # Faz a leitura do arquivo se ele existe
         self.f = open(self.path, 'r')
         texto = self.f.read()
         self.f.close()

         # Primeiro, verifica já foi concluído tudo
         s = r"DESEMPENHO FINAL DO CLASSIFICADOR.*\nTEMPO =.*\nMédia de acerto =.*"
         if not bool( re.search( s, texto ) ):

            # Segundo, verifica se todas as rodadas foram concluídas
            quant_rodadas = len(re.findall( r"Média de acerto =.*", texto))
            if quant_rodadas >= 1 and quant_rodadas < MAX_RODADAS:
               self.status = Status.CONCLUIR_RODADAS

            # Terceiro, verifica se validar os dados
            elif  bool( re.search( r"MG.*", texto ) ):
               self.status = Status.FALTA_VALIDAR
            else:
               self.status = Status.CALCULAR_PARAMETROS
      except IOError: # Cria o arquivo para salvar resultados
         print("Criando o arquivo ", self.path )
         # Criando um objeto do tipo file
         self.f = open(self.path, 'w')
         self.f.close()
            
   # Adicionando dados no arquivo
   def write(self, result):
      self.f = open(self.path, 'a')
      print(result)
      self.f.write("%s\n" % result)
      self.f.close()

   def get_texto(self):
      self.f = open(self.path, 'r')
      texto = self.f.read()
      self.f.close()
      return texto

   def get_directory_main(self):
      return self.directory

   def get_rodadas_finalizadas(self):
      return len(re.findall( r"Média de acerto =.*", self.get_texto()))

   def get_medias(self):
      lista_acerto = re.findall( r"Média.*", self.get_texto())
      medias_acertos = []

      for line in lista_acerto:
         v = float(re.split("=", line.strip() )[1])
         medias_acertos.append( v )

      return medias_acertos

   def get_tempos(self):
      lista_tempo = re.findall( r"TEMPO.*", self.get_texto())
      tempos = []

      for line in lista_tempo:
         v = float(re.split("=", line.strip() )[1])
         tempos.append( v )

      return tempos

   def create_directory(self, directory):

      if( os.path.isdir(directory) ):
         try:
            os.rmdir(directory)
         except OSError:
            print ("Deletion of the directory %s failed" % directory)
         else:
            print ("Successfully deleted the directory %s" % directory)

      access_rights = 0o755

      try:
         os.mkdir(directory, access_rights)
      except OSError:
         print ("Creation of the directory %s failed" % directory)
      else:
         print ("Successfully created the directory %s" % directory)

    

        
