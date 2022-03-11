# ares_awesizer

Who would ever think of extracting 'green energy' on the 'Red Planet'? Well, our DSE Group 23 (graduated 2020) did! 

In place of a personal BSc thesis, TU Delft has students enter 10-member teams and work full-time on a Design Synthesis Exercise. Our challenge was to build a renewable energy system for a ['Rhizome' Mars habitat](http://www.roboticbuilding.eu/project/rhizome-development-of-an-autarkic-design-to-robotic-production-and-operation-system-for-building-off-earth-habitats/), under development for an ESA competition. The system's requirements were very stringent on the overall system mass, and over 50% of the energy made must have been from wind. I was responsible for the design and systems engineering of the airborne wind energy (AWE) pumping power kite system, which was the main focus of the project. A part of this was responsibility was producing a Python software tool based on published physical models for the sizing and optimisation of the power kite with respect to the overall system.

### Final report, published paper, media publications
[Final report](/thesis_material/DSE2020_group_23_Final_Report.pdf)
[Published paper](https://doi.org/10.7480/spool.2021.2.6058)

[New Scientist: Enormous kites flown by robots could help power a Mars colony](https://lnkd.in/degKFAP)
[BBC: Could kite-flying robots power life on Mars?](https://lnkd.in/ddakpHy)
[Daily Mail: Gigantic 530 sq-ft kites flown by robots could be used to harness Mars's strong winds and power human colonies](https://lnkd.in/d6H32Sc)
[Popular Mechanics: Kite-Flying Robots Could Generate Energy on Mars](https://lnkd.in/dUAM6Qx)

Much of the code was pair-coded by me and Daan Witte as the navigator. I was responsible for the main kite operation model, Daan coded the tether sizing model. Siri Heidweiller helped in these efforts. Fernando Corte Vargas and Bart Klootwijk later on produced a code that included the ground station (motor and generator) model. It was my task to coordinate these efforts on the kite's subsystems, and in the end to consolidate the code into a single sizing tool for the system. Credit also goes to Lora Ouromova, Esmée Terwindt, Francesca van Marion, Márton Géczi and Marcel Kempers, who have also contributed to the project and thus affected the final design. 

Our thanks also goes to our supervisors, Roland Schmehl, Dominic von Terzi, Botchu Jyoti and Camila Brito.
