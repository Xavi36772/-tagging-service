"""
Fase 1: Generación de Dataset Sintético.
Genera pares (sinopsis, [tags]) balanceados usando plantillas + Groq API para variedad.
Uso: python generate_dataset.py [--samples N] [--output-dir dataset] [--api-key KEY]
"""

import json
import os
import argparse
import random
from pathlib import Path

random.seed(42)

# ─── Taxonomía y plantillas ───────────────────────────────────────────────

CHARACTERS = [
    "Alejandro", "Valentina", "Santiago", "Camila", "Mateo", "Isabella",
    "Sebastián", "Sofía", "Benjamín", "Regina", "Gabriel", "Emma",
    "Emilio", "Abril", "Damián", "Renata", "Luciano", "Ximena",
    "Julián", "Mariana", "Diego", "Fernanda", "Pablo", "Lucía",
    "Andrés", "Clara", "Tomás", "Ana", "Martín", "Laura",
    "Iker", "Carmen", "Lucas", "Elena", "Hugo", "Luna", "Leo", "Vega"
]

PLACES = [
    "un pueblo remoto", "una ciudad futurista", "una aldea medieval",
    "un laboratorio subterráneo", "una mansión victoriana", "un desierto olvidado",
    "una isla misteriosa", "un bosque encantado", "una estación espacial",
    "una catedral abandonada", "un mercado bullicioso", "una biblioteca infinita",
    "una prisión de máxima seguridad", "un castillo en ruinas", "una nave interestelar",
    "un reino sumergido", "una academia de élite", "un barrio marginal",
    "una fortaleza en la montaña", "una ciudad amurallada"
]

ADJECTIVES = [
    "misterioso", "oscuro", "brillante", "decadente", "majestuoso",
    "olvidado", "sagrado", "maldito", "próspero", "sombrío",
    "neblinoso", "resplandeciente", "árido", "frío", "inhóspito",
    "fascinante", "aterrador", "vibrante", "silencioso", "caótico"
]

EVENTS = [
    "un descubrimiento inesperado", "una tragedia familiar",
    "una guerra civil", "la llegada de un extraño",
    "un fenómeno sobrenatural", "una traición",
    "un secreto del pasado", "una profecía antigua",
    "un asesinato sin resolver", "una desaparición",
    "un experimento científico", "una invasión",
    "un ritual prohibido", "una maldición ancestral",
    "un robo imposible", "una carta anónima",
    "una pandemia", "un terremoto",
    "la caída de un imperio", "un viaje sin retorno"
]

# ─── Plantillas por categoría de tags ─────────────────────────────────────

TEMPLATES = {
    "Acción": [
        "{personaje} es un exmilitar que debe enfrentar a {villano} en una lucha a muerte por el control de {lugar}. Con {habilidad} como única arma, se adentra en un mundo de violencia y traición donde cada segundo cuenta.",
        "Cuando {evento} sacude {lugar}, {personaje} se convierte en la única esperanza. Armado con {habilidad}, deberá abrirse paso a través de enemigos implacables para salvar a quienes ama.",
        "En {lugar}, {personaje} descubre que {evento} es solo el comienzo. Una carrera contrarreloj lo obliga a usar {habilidad} para evitar una catástrofe que cambiaría el mundo para siempre."
    ],
    "Aventura": [
        "{personaje} emprende un viaje épico hacia {lugar} en busca de {objetivo}. Acompañado por {acompañante}, deberá cruzar territorios peligrosos y resolver acertijos ancestrales.",
        "Una vieja mapa lleva a {personaje} a {lugar}, donde {evento} desencadena una búsqueda que cambiará su destino. Cada paso revela secretos que la historia ha mantenido ocultos.",
        "Tras {evento}, {personaje} decide explorar {lugar}. Lo que comienza como una simple curiosidad se convierte en la aventura más grande de su vida, llena de descubrimientos y peligros."
    ],
    "Romance": [
        "En {lugar}, {personaje} conoce a alguien que cambiará su vida para siempre. Entre {obstáculo} y {obstáculo2}, nace un amor que desafía todas las convenciones sociales.",
        "Después de {evento}, {personaje} se encuentra reconstruyendo su vida en {lugar}. Allí, un encuentro casual con {acompañante} enciende una chispa que ninguno esperaba.",
        "{personaje} y {acompañante} son dos almas destinadas a encontrarse, pero {obstáculo} se interpone en su camino. En {lugar}, deberán luchar por su amor contra todo pronóstico."
    ],
    "Drama": [
        "La vida de {personaje} cambia drásticamente cuando {evento} ocurre en {lugar}. Ahora debe enfrentar sus miedos más profundos y tomar decisiones que definirán su futuro.",
        "En {lugar}, {personaje} lidia con las consecuencias de {evento}. Entre el dolor y la esperanza, descubre una fuerza interior que nunca supo que tenía.",
        "Cuando {evento} sacude los cimientos de {lugar}, {personaje} se ve obligado a confrontar verdades incómodas sobre sí mismo y sobre las personas que lo rodean."
    ],
    "Comedia": [
        "{personaje} nunca imaginó que {evento} lo llevaría a las situaciones más absurdas en {lugar}. Con {acompañante} como cómplice, cada día es una nueva aventura digna de risas.",
        "En {lugar}, {personaje} intenta llevar una vida normal, pero {evento} lo convierte en el centro de un caos hilarante. Entre malentendidos y situaciones disparatadas, descubre que la vida es mejor con humor.",
        "Cuando {evento} ocurre en {lugar}, {personaje} reúne a un grupo peculiar de {acompañante} para resolverlo. Lo que sigue es una serie de eventos cómicos que ninguno podría haber predicho."
    ],
    "Terror": [
        "En {lugar}, {personaje} comienza a experimentar fenómenos aterradores después de {evento}. Pronto descubre que algunas puertas, una vez abiertas, no pueden cerrarse.",
        "La noche en que {evento} ocurrió en {lugar}, {personaje} supo que nada volvería a ser igual. Una presencia antigua y malévola lo acecha, y escapar no será tan sencillo.",
        "{personaje} hereda una propiedad en {lugar} después de {evento}. Pero la casa guarda secretos oscuros, y alguien —o algo— no quiere que sean descubiertos."
    ],
    "Suspenso": [
        "{personaje} recibe una amenaza anónima después de {evento} en {lugar}. Cada paso que da lo acerca más a la verdad, pero también al peligro. ¿Podrá descubrir al culpable antes de que sea demasiado tarde?",
        "En {lugar}, {personaje} se da cuenta de que {evento} no fue un accidente. A medida que investiga, se sumerge en una red de mentiras y engaños donde nadie es quien parece.",
        "Cuando {evento} ocurre en {lugar}, {personaje} se convierte en el principal sospechoso. Ahora debe encontrar al verdadero responsable mientras esquiva a quienes quieren silenciarlo."
    ],
    "Misterio": [
        "{personaje} encuentra una pista extraña en {lugar} que lo lleva a investigar {evento}. Cada respuesta genera nuevas preguntas en este rompecabezas que desafía la lógica.",
        "En {lugar}, {personaje} se topa con {evento} que nadie puede explicar. Decidido a descubrir la verdad, se adentra en un laberinto de secretos y revelaciones sorprendentes.",
        "Cuando {evento} ocurre en {lugar}, todas las evidencias apuntan a lo imposible. {personaje} deberá usar su intelecto para resolver el enigma más desconcertante de su carrera."
    ],
    "Ciencia Ficción": [
        "En el año {anio}, en {lugar}, {personaje} es testigo de {evento} que desafía las leyes de la física. La humanidad se enfrenta a un futuro incierto donde la tecnología y la ética chocan.",
        "Después de {evento}, {personaje} descubre que {lugar} alberga una tecnología capaz de cambiar la realidad. Pero jugar a ser dios tiene consecuencias que nadie anticipó.",
        "En {lugar}, un avance científico permite {evento}. {personaje} debe decidir si el progreso vale el precio, mientras fuerzas poderosas intentan controlar este descubrimiento."
    ],
    "Fantasía": [
        "En {lugar}, {personaje} descubre que posee un poder ancestral que emerge después de {evento}. Criaturas mágicas y reinos olvidados lo esperan en una aventura que definirá el destino del mundo.",
        "Cuando {evento} amenaza con destruir {lugar}, {personaje} debe emprender una búsqueda para encontrar un artefacto legendario. Acompañado por {acompañante}, se adentra en tierras donde la magia es tan real como el peligro.",
        "{personaje} creció escuchando historias sobre {lugar}, pero nunca imaginó que {evento} lo llevaría hasta allí. En un mundo de hechizos y criaturas fantásticas, descubrirá que la magia más poderosa está dentro de él."
    ],
    "Distopía": [
        "En un futuro donde el gobierno controla cada aspecto de la vida en {lugar}, {personaje} comienza a cuestionar el sistema después de {evento}. Unirse a la resistencia podría costarle todo.",
        "Después de {evento}, {lugar} se ha convertido en un estado totalitario donde la individualidad está prohibida. {personaje} lucha por preservar su humanidad en un mundo que exige conformidad.",
        "En {lugar}, la sociedad está dividida en clases rígidas. Cuando {evento} ocurre, {personaje} se convierte en un símbolo de rebelión, inspirando a otros a soñar con un mundo diferente."
    ],
    "Cyberpunk": [
        "En las calles iluminadas por neón de {lugar}, {personaje} es un hacker que descubre {evento}. Entre implantes cibernéticos y corporaciones corruptas, la línea entre humano y máquina se desvanece.",
        "Después de {evento}, {personaje} se somete a mejoras cibernéticas para sobrevivir en {lugar}. Pero cada modificación lo aleja más de su humanidad, y el precio de la tecnología es más alto de lo que imaginaba.",
        "En {lugar}, las megacorporaciones controlan todo. {personaje}, un mercenario digital, se ve envuelto en {evento} que podría derribar el sistema desde dentro."
    ],
    "Realismo Mágico": [
        "En {lugar}, la realidad y la fantasía se entrelazan de formas sutiles. Después de {evento}, {personaje} comienza a ver lo extraordinario en lo cotidiano, descubriendo que el mundo es más mágico de lo que parece.",
        "La vida de {personaje} en {lugar} transcurre con normalidad hasta que {evento} trae consigo un toque de magia que transforma su percepción de la realidad.",
        "En {lugar}, las cosas no siempre son lo que parecen. Cuando {evento} ocurre, {personaje} se da cuenta de que lo imposible puede suceder, y que la magia habita en los detalles más simples."
    ],
    "Histórico": [
        "En el {epoca}, en {lugar}, {personaje} vive los acontecimientos que cambiarían el curso de la historia. Entre batallas y revoluciones, su historia personal se entrelaza con los grandes eventos de su tiempo.",
        "Después de {evento}, {personaje} se encuentra inmerso en los conflictos de {lugar} durante el {epoca}. Cada decisión que toma tiene el peso de la historia misma.",
        "En {lugar}, durante el {epoca}, {personaje} descubre un secreto que podría reescribir los libros de historia. {evento} lo obliga a elegir entre la verdad y la lealtad."
    ],
    "Mitología": [
        "Cuando {evento} ocurre en {lugar}, {personaje} descubre que las leyendas antiguas son reales. Dioses olvidados y criaturas mitológicas caminan entre los mortales, y él tiene un papel crucial en la profecía.",
        "En {lugar}, {personaje} se topa con un artefacto de poder divino después de {evento}. Ahora debe enfrentarse a seres de otras eras mientras intenta comprender su conexión con el mundo mítico.",
        "Las antiguas profecías hablaban de {personaje}, un elegido que surgiría en {lugar} después de {evento}. Los dioses observan mientras el destino se despliega de formas inesperadas."
    ],
    "Apocalíptico": [
        "Cuando {evento} desencadena el fin del mundo, {personaje} debe sobrevivir en {lugar} mientras la civilización se desmorona. En un mundo sin reglas, la humanidad muestra su peor y mejor cara.",
        "Después del apocalipsis, {lugar} es un páramo hostil. {personaje} busca un refugio seguro mientras {evento} continúa cobrando víctimas. La esperanza es el lujo más escaso.",
        "En medio del caos de {evento}, {personaje} se convierte en líder de un grupo de supervivientes en {lugar}. Juntos, deberán encontrar la manera de reconstruir un mundo a partir de las cenizas."
    ],
    "Thriller Psicológico": [
        "{personaje} despierta en {lugar} sin recordar cómo llegó allí. {evento} desencadena una espiral de paranoia donde la realidad y la alucinación se mezclan. ¿Puede confiar en su propia mente?",
        "Después de {evento}, {personaje} comienza a notar patrones perturbadores en {lugar}. Cada descubrimiento lo hunde más en una obsesión que amenaza con consumirlo por completo.",
        "En {lugar}, {personaje} se enfrenta a su mayor enemigo: su propia mente. {evento} activa recuerdos reprimidos que podrían destruirlo, y la línea entre víctima y perpetrador se vuelve borrosa."
    ],
    "Crimen": [
        "En {lugar}, {personaje} investiga {evento} que las autoridades prefieren ignorar. Entre callejones oscuros y testigos silenciosos, descubre una red criminal que llega más alto de lo que imaginaba.",
        "Cuando {evento} ocurre en {lugar}, {personaje} es el único dispuesto a hacer justicia. En un mundo donde la ley tiene lagunas, a veces la verdad debe encontrarse fuera de los procedimientos.",
        "{personaje}, un detective en {lugar}, sigue el rastro de {evento}. Cada pista lo lleva más profundo en un submundo de corrupción y violencia donde la verdad tiene un precio."
    ],
    "Western": [
        "En el salvaje oeste de {lugar}, {personaje} llega buscando un nuevo comienzo. Pero {evento} lo arrastra de vuelta a un mundo de forajidos y duelos al atardecer.",
        "Cuando {evento} amenaza {lugar}, {personaje} desenfunda su revólver para proteger lo que es suyo. En una tierra sin ley, la justicia se escribe con plomo.",
        "{personaje} es un forajido que busca redención en {lugar}. Después de {evento}, tiene la oportunidad de cambiar su pasado, pero el fantasma de sus crímenes lo persigue."
    ],
    "Bélico": [
        "En medio de la guerra en {lugar}, {personaje} lucha por sobrevivir mientras {evento} cambia el rumbo del conflicto. Entre el humo y la pólvora, descubre el verdadero costo de la libertad.",
        "Cuando {evento} estalla en {lugar}, {personaje} se alista para defender su hogar. Pero en el campo de batalla, las líneas entre el bien y el mal se difuminan.",
        "Después de {evento}, {personaje} regresa a {lugar} tratando de recordar quién era antes de la guerra. Las cicatrices visibles e invisibles cuentan una historia que pocos pueden entender."
    ],
    "Superhéroes": [
        "Después de {evento}, {personaje} descubre que tiene habilidades extraordinarias en {lugar}. Con grandes poderes vienen grandes responsabilidades, y el mundo necesita un héroe.",
        "En {lugar}, {personaje} oculta su identidad secreta mientras {evento} amenaza la ciudad. Entre salvar vidas y mantener su anonimato, el peso de ser un héroe es más pesado de lo que parece.",
        "Cuando {evento} ocurre en {lugar}, {personaje} debe unir fuerzas con otros seres extraordinarios para enfrentar una amenaza que ningún héroe puede detener solo."
    ],
    "Steampunk": [
        "En {lugar}, donde el vapor mueve el mundo, {personaje} es un inventor que crea artefactos imposibles. Después de {evento}, sus inventos son la clave para salvar una sociedad al borde del colapso.",
        "En un mundo victoriano alternativo, {lugar} funciona con máquinas de vapor y engranajes. {personaje} descubre {evento} que revela una conspiración que podría destruir el equilibrio de poder.",
        "{personaje} navega los cielos de {lugar} en un dirigible, buscando {evento}. En un mundo de engranajes y corsets, la aventura espera en cada nube."
    ],
    "Space Opera": [
        "En los confines de la galaxia, {personaje} capitanea una nave después de {evento}. En {lugar}, imperios interestelares luchan por el control, y él está en el centro de la tormenta.",
        "Cuando {evento} sacude la federación galáctica, {personaje} se ve envuelto en una guerra interestelar. En {lugar}, civilizaciones alienígenas forjan alianzas y traiciones a escala cósmica.",
        "En {lugar}, un imperio galáctico se expande sin control. {personaje}, un piloto rebelde, descubre {evento} que podría cambiar el equilibrio de poder en toda la galaxia."
    ],
    "Slice of Life": [
        "En {lugar}, {personaje} vive su día a día con pequeñas alegrías y desafíos cotidianos. Después de {evento}, aprende a valorar las cosas simples que hacen que la vida valga la pena.",
        "La rutina de {personaje} en {lugar} se ve interrumpida por {evento}, un pequeño cambio que trae nuevas perspectivas y amistades inesperadas.",
        "En la tranquilidad de {lugar}, {personaje} reflexiona sobre la vida después de {evento}. Entre café y conversaciones, descubre que la felicidad está en los momentos más simples."
    ],
    "Coming of Age": [
        "{personaje} crece en {lugar} y {evento} marca el punto de inflexión en su vida. Entre lecciones aprendidas y errores cometidos, el viaje hacia la adultez está lleno de descubrimientos.",
        "En {lugar}, {personaje} navega los turbulentos años de la adolescencia. Cuando {evento} ocurre, debe encontrar su propia voz y definir quién quiere ser.",
        "Después de {evento}, {personaje} deja atrás la infancia y se enfrenta a un mundo nuevo en {lugar}. Las amistades, los sueños y las decepciones moldean el adulto en el que se convertirá."
    ],
    "LGBTQ+": [
        "En {lugar}, {personaje} se embarca en un viaje de autodescubrimiento después de {evento}. Entre el miedo y la esperanza, aprenderá que el amor propio es el primer paso hacia la libertad.",
        "{personaje} siempre sintió que era diferente en {lugar}. Cuando {evento} ocurre, encuentra el valor para vivir su verdad, rodeado de una comunidad que lo acepta como es.",
        "En {lugar}, {personaje} y {acompañante} desafían las normas sociales después de {evento}. Su historia de amor y resistencia inspira a otros a vivir sin miedo."
    ],
    "Feminismo": [
        "En {lugar}, {personaje} desafía las expectativas impuestas después de {evento}. En una sociedad que intenta silenciarla, alza la voz por ella y por todas las que vendrán después.",
        "Cuando {evento} ocurre en {lugar}, {personaje} se convierte en líder de un movimiento que busca igualdad. Entre obstáculos y victorias, demuestra que la fuerza femenina es imparable.",
        "Después de {evento}, {personaje} funda una comunidad de mujeres en {lugar}. Juntas, derriban barreras y construyen un espacio donde la sororidad es la base de todo."
    ],
    "Filosófico": [
        "En {lugar}, {personaje} se cuestiona el significado de la existencia después de {evento}. Cada respuesta lleva a nuevas preguntas en un viaje intelectual que desafía las certezas establecidas.",
        "Cuando {evento} ocurre, {personaje} busca respuestas en los rincones de {lugar}. Entre diálogos y reflexiones, construye una nueva comprensión del mundo y su lugar en él.",
        "En {lugar}, {personaje} se enfrenta a dilemas morales después de {evento}. La línea entre el bien y el mal se desdibuja, y las preguntas son más importantes que las respuestas."
    ],
    "Religioso": [
        "En {lugar}, {personaje} experimenta {evento} que pone a prueba su fe. Entre la duda y la devoción, busca respuestas en los textos sagrados y en la comunidad que lo rodea.",
        "Después de {evento}, {personaje} se retira a {lugar} en busca de iluminación espiritual. El camino hacia la fe está lleno de pruebas que fortalecen su espíritu.",
        "Cuando {evento} sacude {lugar}, {personaje} se convierte en guía espiritual para quienes han perdido la esperanza. Su fe inquebrantable ilumina incluso los momentos más oscuros."
    ],
    "Humor Negro": [
        "En {lugar}, {personaje} enfrenta {evento} de la manera más políticamente incorrecta posible. Entre el desastre y la carcajada, demuestra que reírse de la tragedia es a veces la única salida.",
        "Cuando {evento} ocurre en {lugar}, {personaje} no puede evitar encontrarle el lado absurdo. Con un humor afilado como navaja, navega un mundo donde lo macabro y lo cómico bailan juntos.",
        "La vida de {personaje} en {lugar} es un desastre tras otro. {evento} es solo el último de una larga lista, pero su capacidad para reírse de la desgracia lo mantiene a flote."
    ],
    "Parodia": [
        "En {lugar}, {personaje} se encuentra en medio de {evento} que parodia todos los clichés del género. Nada es lo que parece y todo es una excusa para la risa.",
        "Cuando {evento} ocurre en {lugar}, {personaje} se da cuenta de que está atrapado en una parodia de las historias que tanto critica. Con irreverencia, decide reescribir las reglas.",
        "{personaje} vive en {lugar}, un mundo que es una copia exagerada y absurda de la realidad. {evento} desencadena una serie de situaciones ridículas que desafían toda lógica."
    ],
    "Infantil": [
        "En {lugar}, {personaje} y sus amigos viven una aventura mágica después de {evento}. Con imaginación y trabajo en equipo, demuestran que no hay problema demasiado grande.",
        "Cuando {evento} ocurre en {lugar}, {personaje} aprende una valiosa lección sobre la amistad y la bondad. A veces, las soluciones más simples son las más poderosas.",
        "{personaje} descubre un secreto en {lugar} después de {evento}. Con la ayuda de {acompañante}, se embarca en una travesía llena de risas, aprendizaje y magia."
    ],
    "Juvenil": [
        "En {lugar}, {personaje} comienza un nuevo capítulo de su vida después de {evento}. Entre nuevos amigos, amores y desafíos académicos, descubre quién es realmente.",
        "{personaje} y su grupo de amigos en {lugar} enfrentan {evento} que pondrá a prueba su lealtad. El instituto, los secretos y las primeras veces marcan esta historia.",
        "Cuando {evento} ocurre en {lugar}, {personaje} debe navegar las complejidades de la adolescencia. Entre el amor, la amistad y las decisiones difíciles, cada día es una aventura."
    ],
    "New Adult": [
        "En {lugar}, {personaje} comienza su vida adulta después de {evento}. Entre responsabilidades, relaciones y sueños por cumplir, descubre que crecer es más complicado de lo que imaginaba.",
        "Después de {evento}, {personaje} se muda a {lugar} para empezar de cero. Allí conoce a personas que desafían sus perspectivas y lo obligan a enfrentar sus miedos más profundos.",
        "En {lugar}, {personaje} lidia con las presiones de la vida adulta mientras {evento} amenaza con desestabilizar todo lo que ha construido. El amor, el trabajo y la identidad se entrelazan."
    ],
    "Poesía": [
        "En {lugar}, {personaje} encuentra la belleza en las palabras después de {evento}. Cada verso es un latido, cada estrofa un respiro en este viaje lírico por las emociones humanas.",
        "La poesía de {personaje} captura la esencia de {lugar} y {evento}. Entre rimas y metáforas, teje una historia que habla directamente al corazón.",
        "Después de {evento}, {personaje} comienza a escribir poemas en {lugar}. Las palabras se convierten en su refugio, y cada poema es un paso hacia la sanación."
    ],
    "Epistolar": [
        "{personaje} encuentra un conjunto de cartas en {lugar} después de {evento}. A través de la correspondencia, descubre una historia de amor, pérdida y secretos que abarca generaciones.",
        "En {lugar}, {personaje} comienza un intercambio de cartas con un desconocido después de {evento}. Entre palabras escritas a mano, nace una conexión que trasciende el tiempo y la distancia.",
        "Cuando {evento} ocurre en {lugar}, {personaje} documenta todo en cartas. Cada misiva es un testimonio de esperanza y desesperación en tiempos turbulentos."
    ],
    "Antología": [
        "En {lugar}, {personaje} reúne las historias de quienes han vivido {evento}. Cada relato es una pieza única de un mosaico que revela la complejidad de la experiencia humana.",
        "Después de {evento}, {personaje} compila una colección de cuentos que suceden en {lugar}. Voces diversas se entrelazan para pintar un retrato multifacético de una comunidad.",
        "En {lugar}, {personaje} descubre que {evento} ha afectado a muchas personas de maneras distintas. Cada capítulo cuenta una historia diferente, unidas por un hilo invisible."
    ],
    "Leyendas Urbanas": [
        "En {lugar}, circula una leyenda urbana sobre {evento}. {personaje} decide investigar y descubre que la verdad detrás del mito es más aterradora de lo que nadie imaginaba.",
        "Cuando {evento} ocurre en {lugar}, {personaje} recuerda las historias que escuchó de niño. Las leyendas urbanas cobran vida y la línea entre el mito y la realidad se desvanece.",
        "{personaje} siempre pensó que las leyendas urbanas eran solo cuentos. Pero en {lugar}, después de {evento}, se da cuenta de que algunos mitos esconden una verdad escalofriante."
    ],
    "Survival": [
        "En {lugar}, {personaje} lucha por sobrevivir después de {evento}. Sin recursos y sin ayuda, cada día es una batalla contra los elementos y contra sí mismo.",
        "Cuando {evento} deja a {personaje} varado en {lugar}, debe usar todo su ingenio y fuerza para mantenerse con vida. La naturaleza es hermosa, pero también implacable.",
        "En {lugar}, {personaje} se enfrenta a condiciones extremas después de {evento}. La voluntad de vivir lo impulsa a superar límites que nunca creyó posibles."
    ],
    "Artes Marciales": [
        "En {lugar}, {personaje} entrena en un arte marcial ancestral. Cuando {evento} ocurre, debe poner a prueba años de disciplina y honor para proteger a quienes ama.",
        "Después de {evento}, {personaje} busca al maestro legendario que vive en {lugar}. Aprenderá que las artes marciales son más que golpes: son un camino de crecimiento interior.",
        "En {lugar}, {personaje} participa en un torneo que es mucho más que una competencia. {evento} revela que el verdadero combate no es contra el oponente, sino contra uno mismo."
    ]
}

# Tags que no están en TEMPLATES directamente pero se asignan como secundarias
SECONDARY_TAGS = [
    "LGBTQ+", "Feminismo", "Filosófico", "Religioso", "Humor Negro",
    "Parodia", "Infantil", "Juvenil", "New Adult", "Poesía",
    "Epistolar", "Antología", "Leyendas Urbanas", "Survival",
    "Artes Marciales", "Steampunk", "Slice of Life", "Coming of Age"
]

# ─── Generación de variantes ──────────────────────────────────────────────

def pick(tag_name: str, key: str) -> str:
    """Selecciona aleatoriamente de las listas según la clave."""
    pools = {
        "personaje": CHARACTERS,
        "acompañante": CHARACTERS,
        "lugar": PLACES,
        "evento": EVENTS,
        "villano": CHARACTERS,
        "habilidad": ["su inteligencia", "su fuerza", "su astucia", "su entrenamiento",
                      "sus reflejos", "su determinación", "su capacidad de improvisación",
                      "su conocimiento del terreno"],
        "objetivo": ["un tesoro perdido", "la cura para una enfermedad", "un artefacto mágico",
                     "la verdad oculta", "la libertad de su pueblo", "un antiguo conocimiento"],
        "obstáculo": ["diferencias sociales", "secretos familiares", "la distancia",
                      "prejuicios culturales", "obligaciones contradictorias",
                      "miedos personales", "terceros manipuladores"],
        "obstáculo2": ["el qué dirán", "la presión familiar", "compromisos previos",
                       "traiciones inesperadas", "malentendidos"],
        "anio": ["2045", "2150", "2300", "1890", "2500", "3000", "2100", "1984 alternativo"],
        "epoca": ["siglo XIX", "Renacimiento", "Edad Media", "Antigua Roma",
                  "Guerra Civil", "Revolución Industrial", "Imperio Otomano",
                  "Antiguo Egipto", "Período Victoriano", "Revolución Francesa"],
    }
    pool = pools.get(key, [tag_name])
    return random.choice(pool)


def generate_synopsis(tags: list[str]) -> str:
    """Genera una sinopsis usando plantillas para el primer tag y complementando con los demás."""
    primary = tags[0]
    template_list = TEMPLATES.get(primary, TEMPLATES["Aventura"])
    template = random.choice(template_list)

    # Construir contexto con todos los tags para dar variedad
    context = {k: pick(primary, k) for k in ["personaje", "lugar", "evento",
                "acompañante", "villano", "habilidad", "objetivo",
                "obstáculo", "obstáculo2", "anio", "epoca"]}

    # Elegir personajes distintos
    acompanante_opts = [c for c in CHARACTERS if c != context["personaje"]]
    context["acompañante"] = random.choice(acompanante_opts)
    villano_opts = [c for c in CHARACTERS if c not in [context["personaje"], context["acompañante"]]]
    context["villano"] = random.choice(villano_opts)

    synopsis = template.format(**context)

    # Agregar párrafo extra basado en tags secundarios para dar longitud
    extras = {
        "Romance": f"Mientras tanto, {context['personaje']} descubre que el amor puede surgir en los lugares más inesperados, y que a veces la persona que menos esperas es la que termina robándote el corazón.",
        "Terror": f"Pero lo que {context['personaje']} no sabe es que hay fuerzas oscuras observando desde las sombras, esperando el momento perfecto para atacar.",
        "Suspenso": f"Con cada paso, {context['personaje']} siente que alguien lo observa. La paranoia se convierte en su única compañera mientras la verdad permanece oculta.",
        "Misterio": f"Las piezas del rompecabezas comienzan a encajar, pero {context['personaje']} sabe que todavía falta la pieza más importante, aquella que lo cambiará todo.",
        "Drama": f"Entre lágrimas y sonrisas, {context['personaje']} aprende que las heridas más profundas también pueden sanar con el tiempo y el apoyo de quienes realmente importan.",
        "Humor Negro": f"{context['personaje']} no puede evitar reírse ante el absurdo de la situación. Después de todo, si no te ríes, lloras.",
        "Filosófico": f"{context['personaje']} se pregunta si realmente existe el libre albedrío o si todo está escrito de antemano. Las respuestas, como siempre, generan más preguntas.",
        "Coming of Age": f"Esta experiencia marca un antes y un después en la vida de {context['personaje']}. Las cicatrices de hoy serán las lecciones de mañana.",
        "Survival": f"La sed y el hambre son compañeras constantes, pero {context['personaje']} se niega a rendirse. La voluntad de vivir es más fuerte que cualquier adversidad.",
        "Apocalíptico": f"El mundo que conocían ya no existe. Ahora solo queda aprender a sobrevivir en las ruinas de lo que una vez fue la civilización.",
        "LGBTQ+": f"En un mundo que no siempre entiende, {context['personaje']} encuentra fuerza en su identidad y en la comunidad que lo acepta sin condiciones.",
        "Feminismo": f"{context['personaje']} entiende que la lucha no es solo individual, sino colectiva. Cada pequeña victoria es un paso hacia la igualdad.",
        "Religioso": f"La fe de {context['personaje']} es puesta a prueba una y otra vez, pero en los momentos de mayor oscuridad, encuentra una luz que lo guía.",
        "Infantil": f"Con una sonrisa y la imaginación como herramientas, {context['personaje']} demuestra que la magia existe para quienes creen en ella.",
        "Ciencia Ficción": f"La tecnología avanza más rápido que la ética, y {context['personaje']} se encuentra en el centro de un debate que definirá el futuro de la humanidad.",
        "Cyberpunk": f"En las profundidades digitales de {context['lugar']}, {context['personaje']} descubre que la información es el arma más poderosa de todas.",
        "Poesía": f"Y en medio del caos, {context['personaje']} encuentra belleza: en una palabra, en un gesto, en el simple acto de existir.",
    }

    # Añadir párrafos extra para que llegue a 2-4 párrafos
    extra_paragraphs = []
    for tag in tags[1:]:
        if tag in extras and len(extra_paragraphs) < 3:
            extra_paragraphs.append(extras[tag])

    if not extra_paragraphs:
        fallback = f"{context['personaje']} sabe que {context['lugar']} guarda secretos que aún no ha descubierto. {context['evento']} es solo el principio de una historia que cambiará su vida para siempre."
        extra_paragraphs.append(fallback)

    for p in extra_paragraphs:
        synopsis += "\n\n" + p

    return synopsis


def generate_entry(taxonomy: list[str], primary_tag: str) -> dict:
    """Genera una entrada completa (synopsis, tags)."""
    # Seleccionar 2-4 tags adicionales
    other_tags = [t for t in taxonomy if t != primary_tag]
    # Asegurar que al menos uno sea del mismo "grupo" temático
    similar = [t for t in other_tags if t in
               ["Acción", "Aventura", "Drama", "Comedia", "Suspenso"]]
    n_extra = random.randint(2, 4)
    selected = [primary_tag]

    # Primero intentar añadir tags relacionados
    if similar and random.random() < 0.6:
        selected.append(random.choice(similar))
        n_extra -= 1

    selected.extend(random.sample(other_tags, min(n_extra, len(other_tags))))

    synopsis = generate_synopsis(selected)
    return {"synopsis": synopsis, "tags": selected}


def main():
    parser = argparse.ArgumentParser(description="Generar dataset sintético")
    parser.add_argument("--samples", type=int, default=1500, help="Número de ejemplos a generar")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Directorio de salida")
    parser.add_argument("--api-key", type=str, default=None, help="API key de Groq (opcional, para variedad extra)")
    parser.add_argument("--taxonomy", type=str, default="taxonomy.json", help="Archivo de taxonomía")
    args = parser.parse_args()

    with open(args.taxonomy, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    print(f"Taxonomía cargada: {len(taxonomy)} etiquetas")

    # Generar balanceado: cada tag como primario ~ samples/len(taxonomy) veces
    n_tags = len(taxonomy)
    per_tag = max(1, args.samples // n_tags)
    entries = []

    for tag in taxonomy:
        for _ in range(per_tag):
            entries.append(generate_entry(taxonomy, tag))

    # Completar hasta samples
    while len(entries) < args.samples:
        tag = random.choice(taxonomy)
        entries.append(generate_entry(taxonomy, tag))

    # Recortar y mezclar
    entries = entries[:args.samples]
    random.shuffle(entries)

    print(f"Generadas {len(entries)} entradas sintéticas")

    # Split train/val (80/20)
    split = int(len(entries) * 0.8)
    train = entries[:split]
    val = entries[split:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train)} | Val: {len(val)}")
    print("Dataset generado exitosamente.")


if __name__ == "__main__":
    main()
