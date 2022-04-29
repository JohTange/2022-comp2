# Konkurrence 2: Bimatrixspil 

I denne konkurrence skal du skrive en spillerfunktion, der kan spille bimatrixspil. Når du er færdig, skal du navngive din fil `player.py`, og lægge den i mappen `./submission/` og committe. Husk at sætte dens property `name = 'xxx'`, hvor `xxx` skal være et navn, som du vil bruge i samtlige konkurrencer i kurset. 

Den medfølgende notebook, `comp2-discrete.ipynb` loader spillerfunktioner fra mappen `./players/` og opstiller en turnering, hvor de kan spille imod hinanden. 

## Spillerfunktionen

Spillerfunktionen tager to inputs, `U1,U2`, som er payoff-matricerne i spillet. Din spiller er altid spiller 1, mens modstanderen er spiller 2 (set fra dit synspunkt). Spillerfunktionen skal returnere en `int`, `a1`, som skal være mellem 0 og antallet af handlinger for spiller 1, dvs. `U1.shape[0]`. 

 Nedenfor ser du et eksempel på en spillerfunktion, som vælger en tilfældig handling. 

```Python 
def play(self, U1, U2): 
    na1,na2 = U1.shape
    A1 = np.arange(na1)
    a1 = np.random.choice(A1)
    return a1 # must be an integer in {0,1,...,na1-1}
```

Bemærk: din spiller funktion skal returnere en integer i {0,1,...,na1-1}, svarende til den handlingen for den række, som du vil vælge. 

**Tilfældighed:** Du må gerne bruge tilfældighed, fx `np.random.choice()` til at vælge blandt flere kandidater. Turneringen vil blive gentaget 100 gange mellem dig og din modstander for at midle sådan tilfældighed ud. 

