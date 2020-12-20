# Red Neuronal Conditional GAN

Red Generativa Adversaria para generar digitos escritos a mano (MNIST)

Autor: M. en C. Mario Xavier Canche Uc, Diciembre 2020, mario.canche@cimat.mx  

Basado en: 
- https://www.cimat.mx/~mrivera/cursos/aprendizaje_profundo/dcgan/dcgan.html  
- https://www.tensorflow.org/tutorials/generative/dcgan

## ¿Qué son las GAN?
Las [redes generativas adversarias](https://arxiv.org/abs/1406.2661) (GANs) son una de las ideas más interesantes de la informática actual. Dos modelos son entrenados simultáneamente por un proceso contradictorio. Un *generador* ("the artist")  ("el artista") aprende a crear imágenes que parecen reales, mientras que un *discriminador* ("el crítico de arte") aprende a diferenciar las imágenes reales de las falsificaciones.

![A diagram of a generator and discriminator](./images/gan1.png)

Durante el entrenamiento, el *generador* mejora progresivamente en la creación de imágenes que parecen reales, mientras que el *discriminador* mejora en distinguirlas. El proceso alcanza el equilibrio cuando el *discriminador* ya no puede distinguir imágenes reales de falsificaciones.
![A second diagram of a generator and discriminator](./images/gan2.png)

Para obtener más información sobre las GAN, recomendamos el curso [Intro to Deep Learning](http://introtodeeplearning.com/).
