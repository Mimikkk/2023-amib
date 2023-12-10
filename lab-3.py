from commands import Command
from commands.utils import invoke
import sims

def main():
  commands: list[Command] = []
  experiments = []
  def next_name():
    nonlocal experiments
    experiments.append(f'wlasne-prawd-{len(experiments) + 1}')
    return experiments[-1]

  params = {
    'f1_xo_propor': 1,
    'f1_smX': 0.05,
    'f1_smJunct': 0.02,
    'f1_smComma': 0.02,
    'f1_smModif': 0.10,
    'f1_nmNeu': 0.05,
    'f1_nmConn': 0.10,
    'f1_nmProp': 0.10,
    'f1_nmWei': 1.00,
    'f1_nmVal': 0.05,
  }
  sims.params(next_name(), params)
  sims.params(next_name(), params | {
    'f1_smX': 1.00,
    'f1_smJunct': 1.00,
    'f1_smComma': 1.00,
    'f1_smModif': 1.00,
    'f1_nmNeu': 1.00,
    'f1_nmConn': 1.00,
    'f1_nmProp': 1.00,
    'f1_nmWei': 1.00,
    'f1_nmVal': 1.00,
  })
  sims.params(next_name(), params | {
    'f1_smX': 0.08,
    'f1_smJunct': 0.08,
    'f1_smComma': 0.08,
    'f1_smModif': 0.15,
    'f1_nmNeu': 0.08,
    'f1_nmConn': 0.15,
    'f1_nmProp': 0.15,
    'f1_nmVal': 0.08,
  })

  def create_command(**kwargs):
    commands.append(
      Command(
        optimization_targets=["velocity"],
        population=100,
        generations=50,
        max_part_count=15,
        max_joint_count=30,
        max_neuron_count=20,
        max_connection_count=30,
        genetic_format='1',
        hall_of_fame_size=1,
        crossover_probability=0,
        verbose=True,
        **kwargs,
      )
    )

  for iteration in range(1, 10 + 1):
    create_command(
      name=f"lab-3/1/results/HoF-vel-prawd-{iteration}",
      sims=[f'eval-allcriteria', "deterministic", 'sample-period-longest', experiments[0]],
    )

    create_command(
      name=f"lab-3/2/results/HoF-vel-prawd-{iteration}",
      sims=[f'eval-allcriteria', "deterministic", 'sample-period-longest', experiments[1]],
    )

    create_command(
      name=f"lab-3/3/results/HoF-vel-prawd-{iteration}",
      sims=[f'eval-allcriteria', "deterministic", 'sample-period-longest', experiments[2]],
    )

    create_command(
      name=f"lab-3/4/results/HoF-vel-prawd-{iteration}",
      sims=[f'eval-allcriteria', "deterministic", 'sample-period-longest', experiments[1]],
      parameter_scheduler_factor=0.95,
      parameter_scheduler_parameters=[
        'f1_smX',
        'f1_smJunct',
        'f1_smComma',
        'f1_smModif',
        'f1_nmNeu',
        'f1_nmConn',
        'f1_nmProp',
        'f1_nmVal',
      ],
    )

  invoke(commands)
  print("Processing is done.")

if __name__ == '__main__':
  main()

# 1. Jak "perfperiod" dąży  do zera, to mamy prędkość tzw. chwilową
#
# 2. Gdy "perfperiod" rośnie, krajobraz przystosowania będzie się zmieniał przechodząc z prędkości chwilowej do prędkości mierzonej przez średnie na coraz to dłuższych odcinkach. Przystosowanie będzie coraz to bardziej skupiać się na utrzymaniu prędkości długoterminowej, a nie krótkoterminowej.
#
# 3. Punkty wspólne krajobrazu mówią, o konstrukcjach, które są równie efektywne w prędkości chwilowej i jak tej długoterminowej. tj. I na krótkim odcinku i na długim utrzymują prędkość.
#
# 4. Krajobrazy różnią się łatwością odkrycia optymalizacyjnego dobrych rozwiązań przez to, że te bardziej chwilowe będą mogły możliwe odkryć łatwe akcje, które są nagradzane prędkością, np. szybko zegnij mięsień, a skoordynuj grupę mięśni w czasie, co skutkuje wyższą prędkością, ale wymaga tego patrzenia dalekosiężnego ( prędkość uśredniona). Trudno jest znaleźć taką konstrukcje, która jest szybka, ale też potrafi utrzymać przez dłuższy czas swoją prędkość.

# Zrobiłem 4 eksperymenty, wybrałem reprezentację f1
#
#
# - 1 domyślne wartości - default
#
# - 2 wszystkie wartości jako 1 - all1
#
# - 3 wszystkie wartości dobrane - static
#
# - 4 wszystkie wartości jako 1, ale dopisałem dynamiczny harmonogram zmiany parametrów - dynamic
#
# Harmonogram zmian parametrów to taki, który zmienia skazane wartości parametrów jako wartość * 0.95 z każdą następną generacją.
#
# Statyczne parametry to:
#
# - 'f1_smX': 0.08,
# - 'f1_smJunct': 0.08,
# - 'f1_smComma': 0.08,
# - 'f1_smModif': 0.15,
# - 'f1_nmNeu': 0.08,
# - 'f1_nmConn': 0.15,
# - 'f1_nmProp': 0.15,
# - 'f1_nmVal': 0.08,
#
# Zawarłem 3 wykresy te co w poprzednim laboratorium.
#
# Z wykresów można zobaczyć, że all1 dawał sobie Czasami bardzo dobrze, ale głównie leżał na podłodze i dostawała konwulsji bodźców. Statyczny i dynamiczny dobór poradził sobie lepiej i stabilniej w czasie. średni wynik ich był lepszy niż domyślnej konfiguracji oraz all1. Harmonogram trochę pomaga przewyższyć przeciętny wynik statycznego doboru parametrów.

# Osiągnięte maksymalne stabilne wartości prędkości:
#
# I.6, Ćwiczenie 1
#
# 0.001433347
# II.1, mutacja tylko wag neuronów i ich własności	0.002579245
# II.1, + mutacja modyfikatora długości stick'ów	0.004621532
# II.1, + mutacje: dodawanie i usuwanie neuronów, sensorów i efektorów	0.005943125