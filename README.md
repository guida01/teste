# Federated Learning com Assinaturas Digitais

Este repositório demonstra um cenário simplificado de *federated learning* (FL) baseado na biblioteca [Flower](https://flower.dev/) que inclui a validação de assinaturas digitais para garantir a integridade dos pesos dos modelos enviados pelos clientes.

## Visão Geral

- **Biblioteca principal:** Flower (`flwr`)
- **Estratégia:** FedAvg personalizada para validar assinaturas digitais
- **Assinaturas:** RSA (2048 bits) usando a biblioteca `cryptography`
- **Modelo local:** Regressão logística implementada com NumPy

Cada cliente FL gera um par de chaves RSA, assina os pesos atualizados do modelo local e envia a assinatura ao servidor Flower. O servidor só agrega atualizações cuja assinatura é válida, protegendo contra *model poisoning* simples.

## Estrutura dos Arquivos

- `fl_scenario/digital_signatures.py` – Funções utilitárias para gerar chaves, assinar e validar atualizações.
- `fl_scenario/model.py` – Implementação de um modelo de regressão logística com NumPy e funções auxiliares de treino/avaliação.
- `fl_scenario/client.py` – Cliente Flower que assina os pesos antes de enviá-los ao servidor.
- `fl_scenario/server.py` – Estratégia Flower `FedAvg` customizada que valida assinaturas digitais.
- `fl_scenario/run_simulation.py` – Script principal que cria dados sintéticos, gera pares de chaves, inicia a simulação e executa algumas rodadas de FL.

## Como Executar

1. Crie um ambiente virtual e instale as dependências:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Execute a simulação federada:

   ```bash
   python -m fl_scenario.run_simulation
   ```

   O script executa dois clientes por 3 rodadas, exibindo métricas de perda/accuracy locais. Apenas atualizações com assinaturas válidas são agregadas.

## Resultado Esperado

Durante a execução, o console exibirá logs informando o andamento do treino local, a validação das assinaturas pelo servidor e o resultado agregado de cada rodada.

> **Observação:** Este é um exemplo educacional e não deve ser utilizado em produção sem considerar mecanismos adicionais de segurança, gestão de chaves e comunicação segura entre clientes e servidor.
