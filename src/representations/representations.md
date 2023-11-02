# Reprezentacje systemów

## Podziały

- Sparametryzowane (parameterized): Representations that consist of a set of values for dimensions of a pre-defined structure. They do not
  have properties of combination, control-flow, or abstraction.

- Otwarte (Open-ended): Representations in which the topology of a design is changeable.
    - Nie generatywne (non-generative): Each representational element of an encoded design can map at most once to an element in a designed
      artifact.
        - Bezpośrednie (direct): The encoded design is essentially the same as the actual design.
        - Nie bezpośrednie (indirect): There's a translation or construction process from the encoding to the actual design.
    - Generatywne (generative): Encoded design can reuse elements in its translation to an actual design through abstraction or iteration.
        - Przez domyśl (implicit): Representations consist of a set of rules that implicitly specify a shape, like through an iterative
          construction process similar to cellular automata.
        - Przez wskazanie (explicit): Representations where a design is explicitly represented by an algorithm for constructing it. Think of
          it as indirect representations with reuse.
