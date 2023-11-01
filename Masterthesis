### Introduction

This file contains both the abstract and the conclusion of my master thesis. It is intended to serve as a brief summary of the master thesis and to link the code evaluation to the purpose of the thesis itself.
If you want furhter information feel free to contact me.
In the future I might come back to this and develop and change the code further in another repro :)


# AI-assisted real time stabilization of an optical resonator.

## Abstract
This thesis examines the application of artificial intelligence to optimize optical resonator mode matching. 
A thorough experimental setup was devised, incorporating essential elements for the Pound-Drever-Hall method and developing a digital twin for AI training. 
Various improvements in AI performance were explored and assessed across different simulated environments. 
While the AI model proved successful in familiar settings by identifying optimal mode matching conditions, it encountered challenges when confronted with unfamiliar environments.
Nevertheless, notable potential lies within integrating AI into this field if adjustments are made to have the agent generate relative actions instead of absolute actions.

## Conclusion
This study aimed to examine the potential use of AI in optimizing the mode
matching of an optical resonator. To achieve this, a comprehensive experimental
setup was carefully designed and assembled. The design of the setup prioritized
integrating all necessary optical and electronic components needed for applying
the Pound-Drever-Hall method. The frequency of the sidebands in the
PDH method was carefully selected to correspond to a non-resonant frequency
of the optical resonator. Specifically, a non-resonant frequency was chosen as the
sideband frequency for PDH to correspond with the target optical resonator. All
optomechanical components, fiber optic components, and electronic elements
were meticulously assembled at LENA’s optical laboratory.
Furthermore, a digital twin of this setup was created to facilitate the training of
artificial intelligence in various simulated environments. This involved modeling
the environment as a combination of a beamwalk and a Fabry-Perot interferometer.
The beamwalk simulation was achieved through the utilization of ABCDmatrices
and q-parameters. Similarly, for the FPI, calculations were performed
using the ABCD-matrix law to determine the q-parameter of the fundamental
mode, allowing for derivation of a mismatch error. Utilizing this mismatch error,
intensities were computed as inputs for the AI agent. Consequently, it generated
outputs representing the overall position integration on the linear stage within
beamwalk simulations.
In order to evaluate the effectiveness of the AI model, extensive studies were
conducted to investigate potential enhancements. These investigations involved
introducing noise into the distances between optical elements and the length of
the FPI within the environment model. Various configurations of lenses in different
environments were also simulated. Furthermore, efforts were made to develop
a universal AI that can achieve optimal mode matching even in unfamiliar environments
without relying on specific training data. The investigations ultimately
concluded with a search for optimal hyperparameters.
In conclusion, the current capabilities of the AI in actively regulating the mode
matching are still limited. Although AI can find optimal solutions in known environments,
similar results can be obtained using simple optimization functions
without neural networks. However, integrating AI into mode matching processes
holds promising potential, so that this research can serve as a solid basis for future
investigations in this area.

## Outlook
In order to implement the AI model effectively in a practical setting, it is imperative
to conduct further research and investigations building upon this study. This
section presents an overview of the necessary modifications that should be made
to the AI system, as well as outlining potential areas for additional investigation
that would yield fruitful results.
In their respective studies, Sorokin et al. (2020) and Makarenko et al. (2021) both
utilize highly-dimensional environmental observations and produce outputs that
reflect relative values rather than absolute values. As a result, the differences from
their work can be considered as potential areas for improvement in this study. To
begin, the agent’s output copuld be modified by reducing the action space from
1000 to five distinct actions. For instance, these actions could include a large positive
step, a small positive step, a small negative step, a large negative step, and an
action representing no change. By employing this approach of relative actions, it
should enable the agent to optimize its performance across various environments
effectively.
Moreover, it would be prudent to modify the input data for the agent. For instance,
incorporating various environment parameters such as the mirror’s curvature
radius, distance to the FPI, linear stage position, and collimator values could
enhance performance. Additionally, exploring potential benefits by altering network
architecture from a Feedforward Neural Network to a Recurrent Neural Network
may yield positive results. Similarly, considering changing input representation
from CCD camera images and adapting neural networks accordingly into
Convolutional Neural Networks is worth investigating.
Lastly, it would be of interest to investigate whether expanding the agent’s action
space with higher-dimensional options, such as employing controllable nonlinear
lenses, as demonstrated in previous research by Tarquin Ralph et al. (2020),
would enable the agent to adapt its actions based on specific environmental conditions.
