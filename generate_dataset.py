"""
Generates a diverse synthetic dataset for multi-label tag classification.
Uses expanded templates with high structural variety to force the model
to learn word-tag associations rather than memorizing patterns.

Usage: python generate_dataset.py [--samples N] [--output-dir dataset]
"""

import json
import random
from pathlib import Path

random.seed(42)

# ─── Expanded Vocabulary ─────────────────────────────────────────────────

CHARACTERS = [
    "Alejandro", "Valentina", "Santiago", "Camila", "Mateo", "Isabella",
    "Sebasti\u00e1n", "Sof\u00eda", "Benjam\u00edn", "Regina", "Gabriel", "Emma",
    "Emilio", "Abril", "Dami\u00e1n", "Renata", "Luciano", "Ximena",
    "Juli\u00e1n", "Mariana", "Diego", "Fernanda", "Pablo", "Luc\u00eda",
    "Andr\u00e9s", "Clara", "Tom\u00e1s", "Ana", "Mart\u00edn", "Laura",
    "Iker", "Carmen", "Lucas", "Elena", "Hugo", "Luna", "Leo", "Vega",
    "David", "Paula", "Daniel", "Marta", "Carlos", "Rosa", "Jorge", "Iris"
]

VOCATIONS = [
    "un m\u00e9dico", "una maestra", "un periodista", "una arquitecta",
    "un soldado retirado", "una bibliotecaria", "un abogado", "una cient\u00edfica",
    "un marinero", "una bailarina", "un herrero", "una costurera",
    "un mercader", "una sanadora", "un escriba", "una guerrera",
    "un campesino", "una princesa", "un monje", "una exploradora",
    "un detective", "una piloto", "un ingeniero", "una m\u00fasica",
    "un cocinero", "una pintora", "un carpintero", "una astr\u00f3noma",
    "un capit\u00e1n", "una cart\u00f3grafa", "un alquimista", "una reina",
    "un ladr\u00f3n", "una esp\u00eda", "un cazador", "una profetisa",
    "un navegante", "una druida", "un mercenario", "una embajadora",
    "un poeta", "una escultora", "un arquero", "una jinete",
]

PLACES = [
    "un pueblo remoto en las monta\u00f1as", "una ciudad futurista",
    "una aldea medieval amurallada", "un laboratorio subterr\u00e1neo",
    "una mansi\u00f3n victoriana abandonada", "un desierto olvidado por Dios",
    "una isla misteriosa en el Pac\u00edfico", "un bosque encantado",
    "una estaci\u00f3n espacial en \u00f3rbita", "una catedral g\u00f3tica",
    "un mercado bullicioso", "una biblioteca infinita sin fin aparente",
    "una prisi\u00f3n de m\u00e1xima seguridad", "un castillo en ruinas",
    "una nave interestelar", "un reino sumergido bajo el oc\u00e9ano",
    "una academia de \u00e9lite", "un barrio marginal de la periferia",
    "una fortaleza en la cima de una monta\u00f1a", "una ciudad amurallada",
    "un santuario oculto en la selva", "una mansi\u00f3n embrujada",
    "un orfanato abandonado", "una base militar secreta",
    "un laboratorio de investigaci\u00f3n", "un palacio de hielo",
    "un tren transcontinental", "una isla desierta",
    "un campo de refugiados", "una cripta subterr\u00e1nea",
    "una torre de vigilancia", "un hospital abandonado",
    "una f\u00e1brica en desuso", "un circo ambulante",
    "un monasterio remoto", "una cueva desconocida",
    "un asentamiento en Marte", "una ciudad subterr\u00e1nea",
    "un barco a la deriva", "un reino en las nubes",
]

EVENTS = [
    "un descubrimiento inesperado", "una tragedia familiar",
    "una guerra civil devastadora", "la llegada de un misterioso extra\u00f1o",
    "un fen\u00f3meno sobrenatural inexplicable", "una traici\u00f3n inesperada",
    "un secreto del pasado que sale a la luz", "una profec\u00eda antigua que se cumple",
    "un asesinato sin resolver", "una desaparici\u00f3n repentina",
    "un experimento cient\u00edfico que sale mal", "una invasi\u00f3n extraterrestre",
    "un ritual prohibido", "una maldici\u00f3n ancestral",
    "un robo imposible de resolver", "una carta an\u00f3nima amenazante",
    "una pandemia mortal", "un terremoto devastador",
    "la ca\u00edda de un imperio", "un viaje del que no hay retorno",
    "una erupci\u00f3n volc\u00e1nica", "el nacimiento de un elegido",
    "una sequ\u00eda que dura d\u00e9cadas", "un naufragio en alta mar",
    "una invasi\u00f3n militar", "el descubrimiento de un portal dimensional",
    "una revuelta popular", "una plaga que afecta los cultivos",
    "una alianza inesperada", "un eclipse que marca el destino",
    "la llegada de un cometa", "un golpe de estado",
    "una epidemia de violencia", "el despertar de una entidad dormida",
    "una serie de accidentes extra\u00f1os", "un secuestro",
    "una declaraci\u00f3n de guerra", "el hallazgo de un mapa antiguo",
    "una tormenta mortal", "el colapso de un reactor",
]

MOTIVATIONS = [
    "salvar a su familia", "proteger su hogar", "encontrar la verdad",
    "descubrir qui\u00e9n es realmente", "demostrar su val\u00eda",
    "vengar una injusticia", "recuperar lo que perdi\u00f3",
    "escapar de un destino tr\u00e1gico", "cumplir una promesa",
    "encontrar un tesoro legendario", "liberar a su pueblo",
    "romper una maldici\u00f3n", "desentra\u00f1ar un enigma",
    "cambiar el curso de la historia", "proteger un secreto",
    "sanar una herida del pasado", "encontrar la paz interior",
    "reunirse con un ser querido", "luchar por la justicia",
    "construir un futuro mejor", "desafiar las normas establecidas",
    "preservar la memoria de su cultura", "encontrar esperanza",
    "recuperar la confianza perdida", "enfrentar sus propios miedos",
]

CONFLICTS = [
    "el gobierno totalitario", "una corporaci\u00f3n despiadada",
    "un ser maligno ancestral", "sus propios demonios internos",
    "una sociedad que no lo acepta", "el paso del tiempo",
    "un rival implacable", "la naturaleza salvaje",
    "un secreto que podr\u00eda destruirlo todo", "la pobreza y la desigualdad",
    "sus propias limitaciones", "una maldici\u00f3n hereditaria",
    "la burocracia corrupta", "una organizaci\u00f3n criminal",
    "el fanatismo religioso", "un virus mortal",
    "la codicia humana", "la ignorancia y el miedo",
    "una familia que lo rechaza", "las diferencias culturales",
    "la presi\u00f3n social", "un antiguo dios despierto",
    "el cambio clim\u00e1tico", "la guerra sin fin",
    "una inteligencia artificial hostil", "realidades alternas",
]

ADJECTIVES = [
    "misterioso", "oscuro", "brillante", "decadente", "majestuoso",
    "olvidado", "sagrado", "maldito", "pr\u00f3spero", "sombr\u00edo",
    "neblinoso", "resplandeciente", "\u00e1rido", "helado", "inh\u00f3spito",
    "fascinante", "aterrador", "vibrante", "silencioso", "ca\u00f3tico",
    "melanc\u00f3lico", "radiante", "polvoriento", "encantado", "mortal",
    "g\u00e9lido", "h\u00famedo", "infinito", "olvidado", "sagrado",
]

WEATHER = [
    "bajo una lluvia torrencial", "en una noche sin luna",
    "con el sol abrasador del mediod\u00eda", "durante una tormenta el\u00e9ctrica",
    "en medio de una niebla espesa", "bajo la luz p\u00e1lida de la luna",
    "con el viento aullando entre los \u00e1rboles", "en un d\u00eda de calor sofocante",
    "durante una nevada implacable", "en un amanecer de oto\u00f1o",
    "bajo un cielo de ceniza", "en la oscuridad de la medianoche",
    "con una brisa suave del mar", "en medio de una tormenta de arena",
    "en un d\u00eda nublado y gris", "bajo la lluvia \u00e1cida de la ciudad",
    "en la penumbra del atardecer", "durante una aurora boreal",
    "con el cielo despejado y estrellado", "en una ma\u00f1ana de primavera",
]

TRANSITIONS_SIMPLE = [
    "Sin embargo,", "Adem\u00e1s,", "Pero", "A pesar de todo,",
    "Mientras tanto,", "Por otro lado,", "Sin embargo,",
    "Entonces,", "Pero las cosas cambian cuando",
    "Sin embargo, nada es lo que parece cuando",
    "Pero el destino tiene otros planes.",
    "Sin embargo, el peligro acecha en cada esquina.",
    "Hasta que un d\u00eda,",
    "Pero todo se complica cuando",
]

TRANSITIONS_NARRATIVE = [
    "Lo que ignora es que",
    "Lo que no sabe es que",
    "El problema es que",
    "La cuesti\u00f3n es que",
    "Lo peor de todo es que",
    "Pero lo que parec\u00eda sencillo se complica cuando",
    "La verdad es mucho m\u00e1s oscura:",
    "Nadie imagina que detr\u00e1s de todo se esconde",
    "El destino le tiene preparada una sorpresa:",
]

EMOTIONS = [
    "con el coraz\u00f3n encogido por el miedo",
    "lleno de esperanza",
    "consumido por la culpa",
    "con una determinaci\u00f3n inquebrantable",
    "con el alma rota en pedazos",
    "ardiendo de ira contenida",
    "con la certeza de quien no tiene nada que perder",
    "temblando de incertidumbre",
    "con una calma que sorprende incluso a s\u00ed mismo",
    "desbordado por la tristeza",
    "con una sonrisa que esconde el dolor",
    "con la mirada perdida en el horizonte",
    "con el coraz\u00f3n dividido entre el deber y el deseo",
]


def pick(key: str) -> str:
    pools = {
        "personaje": CHARACTERS, "acompa\u00f1ante": CHARACTERS,
        "lugar": PLACES, "evento": EVENTS,
        "vocacion": VOCATIONS, "motivacion": MOTIVATIONS,
        "conflicto": CONFLICTS, "adjetivo": ADJECTIVES,
        "clima": WEATHER, "emocion": EMOTIONS,
        "villano": CHARACTERS,
    }
    pool = pools.get(key, CHARACTERS)
    return random.choice(pool)


def pick_distinct(*keys: str) -> dict:
    """Pick distinct characters for each key."""
    chars = CHARACTERS.copy()
    random.shuffle(chars)
    result = {}
    idx = 0
    for key in keys:
        if key in ("personaje", "acompa\u00f1ante", "villano"):
            result[key] = chars[idx]
            idx += 1
        else:
            result[key] = pick(key)
    return result


def maybe(probability=0.5):
    return random.random() < probability


def wrap(synopsis: str) -> str:
    """Add random structural variation to a base synopsis."""
    parts = [synopsis]

    if maybe(0.3):
        parts.insert(0, f"{pick('clima')}, ")

    if maybe(0.25):
        conj = random.choice([
            f" {pick('personaje')} sabe que el tiempo se acaba.",
            f" Pero {pick('personaje')} no est\u00e1 dispuesto a rendirse.",
            f" {pick('personaje')} tendr\u00e1 que enfrentarse a {pick('conflicto')}.",
            f" La {pick('motivacion')} lo impulsa a seguir adelante.",
        ])
        parts.append(conj.strip())

    if maybe(0.2):
        emotion = f"\n\n{pick('emocion')}, " + pick("personaje") + " se prepara para lo peor."
        parts.append(emotion)

    return "".join(parts)


# ─── Templates: 15-20 per tag ──────────────────────────────────────────

T = {
"Acci\u00f3n": [
    # Character-start
    "{p} es {v} que debe enfrentar a {v2} en una lucha a muerte por el control de {l}.",
    "{p}, {v}, recibe una misi\u00f3n imposible: infiltrarse en {l} y detener {e}.",
    "{p} ha entrenado toda su vida para este momento. {e} es la oportunidad que esperaba para demostrar su val\u00eda en {l}.",
    # Event-start
    "Cuando {e} sacude {l}, {p} es la \u00fanica persona capaz de hacer algo al respecto. Armado hasta los dientes, se adentra en territorio enemigo.",
    "{e} desencadena una espiral de violencia en {l}. {p}, {v}, se ve arrastrado a un conflicto que amenaza con consumirlo todo.",
    # Place-start
    "En {l} reina el caos despu\u00e9s de {e}. {p} debe abrirse paso entre enemigos para {m}.",
    "Las calles de {l} son un campo de batalla. {p} lucha por sobrevivir mientras {e} se intensifica.",
    # Question-start
    "\u00bfPuede {p} detener {e} antes de que {l} caiga en manos equivocadas? El tiempo corre y cada segundo cuenta.",
    # Description-start
    "Un enfrentamiento \u00e9pico est\u00e1 por ocurrir en {l}. {p} y {v2} se miden en un duelo que decidir\u00e1 el destino de todos.",
    # Dialogue-start
    "\u00abNo hay vuelta atr\u00e1s\u00bb, dice {p} mientras prepara sus armas. {e} ha comenzado, y {l} nunca volver\u00e1 a ser el mismo.",
    # More varied
    "La misi\u00f3n de {p} es clara: llegar a {l}, neutralizar la amenaza y evitar {e}. Pero nada sale seg\u00fan lo planeado.",
    "{p} pertenece a una unidad de \u00e9lite enviada a {l} cuando {e} ocurre. La guerra no espera a nadie.",
    "El \u00fanico camino para {m} pasa por {l}. {p} sabe que no todos sobrevivir\u00e1n al intento.",
    "Entre disparos y explosiones, {p} intenta {m} en medio del conflicto que {e} ha desatado en {l}.",
    "{p} descubre que {e} es solo la punta del iceberg. En {l} se esconde una amenaza mucho mayor que requiere acci\u00f3n inmediata.",
],
"Aventura": [
    "{p} emprende un viaje \u00e9pico hacia {l} en busca de {m}.",
    "Un viejo mapa lleva a {p} a {l}, donde {e} desencadena una b\u00fasqueda extraordinaria.",
    "Tras {e}, {p} decide explorar {l}. La curiosidad se convierte en la aventura m\u00e1s grande de su vida.",
    "En {l}, {p} encuentra una pista que podr\u00eda llevar al mayor descubrimiento de la historia.",
    "La br\u00fajula de {p} se\u00f1ala hacia {l}, un lugar que pocos han visto. {e} es solo el comienzo.",
    "\u00bfQu\u00e9 secretos esconde {l}? {p} est\u00e1 decidido a descubrirlos, cueste lo que cueste.",
    "Cuando {e} ocurre, {p} sabe que su vida rutinaria ha terminado. La llamada de la aventura lo espera en {l}.",
    "{p} nunca pidi\u00f3 una aventura, pero {e} lo empuja a cruzar oc\u00e9anos y monta\u00f1as hasta llegar a {l}.",
    "Un desconocido entrega a {p} un objeto misterioso. La \u00fanica pista lleva a {l}, donde le espera {e}.",
    "Junto a {a}, {p} navega hacia lo desconocido. {l} promete maravillas y peligros m\u00e1s all\u00e1 de toda imaginaci\u00f3n.",
    "La \u00faltima voluntad de un ser querido env\u00eda a {p} a {l} en busca de respuestas.",
    "Expediciones anteriores a {l} nunca regresaron. {p} est\u00e1 a punto de descubrir por qu\u00e9.",
    "Cada paso en {l} revela maravillas que la civilizaci\u00f3n ha olvidado. {e} es la llave para desbloquear el misterio.",
    "Lo que comienza como una excursi\u00f3n en {l} se convierte en una carrera contrarreloj cuando {e} ocurre.",
    "El esp\u00edritu aventurero de {p} no lo deja ignorar {e}. Su destino lo espera en {l}.",
],
"Romance": [
    "En {l}, {p} conoce a alguien que cambiar\u00e1 su vida. Entre {c} nace un amor prohibido.",
    "Despu\u00e9s de {e}, {p} reconstruye su vida en {l}. All\u00ed, un encuentro casual enciende una chispa inesperada.",
    "{p} y {a} son almas gemelas separadas por {c}. En {l}, el destino les da una segunda oportunidad.",
    "El amor florece en los lugares m\u00e1s inesperados. {p} descubre esto cuando llega a {l} tras {e}.",
    "\u00bfPuede el amor sobrevivir a {c}? {p} y {a} est\u00e1n a punto de averiguarlo en {l}.",
    "{p} jura que nunca volver\u00e1 a enamorarse. Pero {a} aparece en {l} y todos sus planes se desmoronan.",
    "{e} re\u00fane a {p} y {a} despu\u00e9s de a\u00f1os separados. En {l}, las viejas heridas y los viejos sentimientos afloran.",
    "Una carta extraviada, un malentendido, y el coraz\u00f3n de {p} queda atrapado entre el deber y el deseo en {l}.",
    "En {l}, {p} y {a} son rivales. Pero el amor no entiende de bandos, y {e} los obliga a trabajar juntos.",
    "El matrimonio arreglado de {p} se complica cuando conoce a {a} en {l}. {e} lo cambia todo.",
    "Bajo las estrellas de {l}, {p} confiesa su amor. Pero {c} amenaza con separarlos para siempre.",
    "La dulce melancol\u00eda de {l} envuelve a {p}, que a\u00fan no supera {e}. Hasta que {a} llega a su vida.",
    "Un viaje a {l} era justo lo que {p} necesitaba. Nunca imagin\u00f3 que tambi\u00e9n encontrar\u00eda el amor.",
    "{p} siempre crey\u00f3 en el amor, pero {e} destruy\u00f3 esa fe. En {l}, {a} lo ayudar\u00e1 a creer de nuevo.",
    "Dos extra\u00f1os, {p} y {a}, comparten un tren hacia {l}. {e} los une de una manera que ninguno esperaba.",
],
"Comedia": [
    "{p} nunca imagin\u00f3 que {e} lo llevar\u00eda a situaciones tan absurdas en {l}.",
    "En {l}, {p} intenta tener un d\u00eda normal, pero {e} convierte todo en un caos hilarante.",
    "La vida de {p} es un desastre, y {e} es la guinda del pastel. Por suerte, en {l} encuentra a {a}, su c\u00f3mplice perfecto.",
    "\u00bfQu\u00e9 podr\u00eda salir mal? Todo, cuando {p} decide mudarse a {l} justo despu\u00e9s de {e}.",
    "El plan de {p} era sencillo, pero en {l} nada es sencillo. Y entonces lleg\u00f3 {e}.",
    "{p} tiene un talento especial para meterse en problemas. En {l}, ese talento alcanza nuevas cotas.",
    "Cuando {e} ocurre en {l}, {p} re\u00fane a un grupo peculiar para resolverlo. el caos est\u00e1 garantizado.",
    "Entre malentendidos y situaciones disparatadas, {p} descubre que la vida es mejor con humor.",
    "{p} jura que esta vez ser\u00e1 responsable. Pero {a} lo convence de lo contrario, y terminan en {l} viviendo {e}.",
    "La suegra de {p} llega de visita justo cuando {e} ocurre en {l}. Las risas est\u00e1n aseguradas.",
    "Una herencia inesperada lleva a {p} a {l}, donde descubre que la fortuna familiar es... peculiar.",
    "{p} decide cambiar su vida y emprende un negocio en {l}. Por supuesto, todo sale mal desde el principio.",
    "El d\u00eda m\u00e1s importante de {p} coincide con {e}. El universo tiene un sentido del humor muy retorcido.",
    "La familia de {p} es disfuncional, y una reuni\u00f3n en {l} despu\u00e9s de {e} lo demuestra de forma hilarante.",
    "{p} y {a} son vecinos en {l}. Una guerra de ruidos se convierte en algo m\u00e1s cuando {e} ocurre.",
],
"Drama": [
    "La vida de {p} cambia dr\u00e1sticamente cuando {e} ocurre. En {l}, deber\u00e1 enfrentar sus miedos m\u00e1s profundos.",
    "En {l}, {p} lidia con las consecuencias de {e}. Entre el dolor y la esperanza, encuentra una fuerza interior desconocida.",
    "Cuando {e} sacude {l}, {p} se enfrenta a verdades inc\u00f3modas sobre s\u00ed mismo y quienes lo rodean.",
    "{p} lo ten\u00eda todo, pero {e} se lo arrebata. Ahora en {l}, debe reconstruir su vida desde cero.",
    "Las decisiones del pasado persiguen a {p}. En {l}, {e} lo obliga a enfrentar fantasmas que cre\u00eda enterrados.",
    "Una madre, un hijo, un secreto guardado durante a\u00f1os. {e} en {l} obliga a {p} a enfrentar la verdad.",
    "El silencio de {l} envuelve a {p}, que busca respuestas despu\u00e9s de {e}. Las encuentra, pero no son las que esperaba.",
    "Entre l\u00e1grimas y sonrisas, {p} aprende que las heridas m\u00e1s profundas sanan con tiempo y apoyo.",
    "{p} debe tomar una decisi\u00f3n imposible en {l}: {m}, sabiendo que cualquier elecci\u00f3n tendr\u00e1 un precio.",
    "La enfermedad, la p\u00e9rdida y la redenci\u00f3n se entrelazan en la historia de {p} en {l}. {e} es el punto de inflexi\u00f3n.",
    "Cada persona en {l} guarda una historia. {p} descubre la suya propia cuando {e} ocurre.",
    "El regreso de {p} a {l} despu\u00e9s de {e} no es el reencuentro feliz que imaginaba.",
    "La culpa consume a {p}. En {l}, buscar\u00e1 redenci\u00f3n, pero {e} le recuerda que algunas cicatrices no sanan.",
    "{p} siempre fue fuerte, pero {e} en {l} pone a prueba los l\u00edmites de su resistencia.",
    "Todo lo que {p} cre\u00eda saber sobre su familia se desmorona cuando {e} ocurre en {l}.",
],
"Terror": [
    "En {l}, {p} comienza a experimentar fen\u00f3menos aterradores despu\u00e9s de {e}. Algunas puertas no deber\u00edan abrirse.",
    "La noche en que {e} ocurre en {l}, {p} sabe que nada volver\u00e1 a ser igual. Una presencia mal\u00e9vola lo acecha.",
    "{p} hereda una propiedad en {l} tras {e}. La casa guarda secretos oscuros que claman por ser descubiertos.",
    "El horror acecha en {l}. {p} lo descubre cuando {e} despierta algo que debi\u00f3 permanecer dormido.",
    "Hay algo malo en {l}. Los habitantes lo saben, pero nadie habla. {p} lo descubrir\u00e1 por las malas.",
    "Los sue\u00f1os de {p} se vuelven pesadillas despu\u00e9s de {e}. Y las pesadillas est\u00e1n cobrando vida en {l}.",
    "Una niebla espesa cubre {l}. Dentro de ella, hay algo que caza. {p} ser\u00e1 su pr\u00f3xima presa.",
    "{p} encuentra un diario antiguo en {l}. Cada p\u00e1gina describe {e} con escalofriante detalle.",
    "Las sombras en {l} se mueven. {p} quiere creer que es su imaginaci\u00f3n, pero {e} demuestra lo contrario.",
    "El \u00faltimo habitante de {l} advierte a {p}: \u00abVete antes de que {e} te encuentre\u00bb.",
    "No hay escapatoria de {l}. {p} lo comprende cuando {e} ocurre y todas las salidas desaparecen.",
    "Cada noche, {p} escucha pasos en {l}. {e} se acerca, y no hay lugar donde esconderse.",
    "El culto en {l} ha estado esperando a {p} mucho antes de que naciera. {e} es la se\u00f1al que esperaban.",
    "Lo que {p} vio en {l} no puede explicarse con la l\u00f3gica. {e} trasciende la comprensi\u00f3n humana.",
    "El espejo en {l} no refleja lo que deber\u00eda. Cuando {p} lo mira, ve {e} suceder una y otra vez.",
],
"Suspenso": [
    "{p} recibe una amenaza an\u00f3nima tras {e}. Cada paso lo acerca a la verdad y al peligro.",
    "En {l}, {p} descubre que {e} no fue un accidente. Una red de mentiras y enga\u00f1os lo envuelve.",
    "Cuando {e} ocurre, {p} se convierte en el principal sospechoso. Debe encontrar al culpable mientras lo persiguen.",
    "Alguien observa cada movimiento de {p} en {l}. La paranoia crece y {e} se acerca inexorablemente.",
    "El tiempo corre para {p}. Tiene 48 horas para resolver {e} en {l} antes de que sea demasiado tarde.",
    "La tensi\u00f3n en {l} es insoportable. {p} sabe que {e} ocurrir\u00e1 pronto, pero no sabe cu\u00e1ndo ni d\u00f3nde.",
    "{p} descubre una grabaci\u00f3n que cambiar\u00eda todo. Pero en {l}, la verdad tiene un precio y {e} es solo el primer pago.",
    "Cada persona en {l} miente. {p} debe descubrir qui\u00e9n dice la verdad sobre {e} antes de que sea v\u00edctima del enga\u00f1o.",
    "La cuenta regresiva ha comenzado. {p} corre contra el reloj en {l} para evitar {e}.",
    "Un juego del gato y el rat\u00f3n en {l}. {p} es la presa, y alguien muy cercano es el cazador.",
    "La confianza es un lujo que {p} no puede permitirse en {l}. {e} ha demostrado que cualquiera puede ser culpable.",
    "Las pistas llevan a {p} cada vez m\u00e1s profundo en un misterio que amenaza con consumirlo.",
    "\u00bfQui\u00e9n est\u00e1 detr\u00e1s de {e}? {p} sigue el rastro en {l}, pero cada respuesta genera m\u00e1s preguntas.",
    "El silencio de los testigos en {l} es ensordecedor. {p} presiona para obtener respuestas sobre {e}.",
    "Tres personas saben la verdad sobre {e} en {l}. Dos est\u00e1n muertas. {p} es la tercera.",
],
"Misterio": [
    "{p} encuentra una pista en {l} que lo lleva a investigar {e}. Cada respuesta genera nuevas preguntas.",
    "En {l}, {p} se topa con {e}, un suceso que nadie puede explicar. Descubrir la verdad se vuelve su obsesi\u00f3n.",
    "Cuando {e} ocurre en {l}, las evidencias apuntan a lo imposible. {p} deber\u00e1 usar su intelecto para resolverlo.",
    "Un objeto extra\u00f1o aparece en {l}. {p} investiga su origen y descubre que est\u00e1 conectado con {e}.",
    "Diez personas en {l}, una sola verdad sobre {e}. {p} deber\u00e1 separar los hechos de las mentiras.",
    "El caso parece cerrado, pero {p} nota inconsistencias. {e} en {l} esconde m\u00e1s de lo que parece a simple vista.",
    "La \u00fanica testigo de {e} en {l} ha desaparecido. {p} sigue su rastro a trav\u00e9s de un laberinto de secretos.",
    "Un enigma matem\u00e1tico, un c\u00f3digo oculto, y {e} en {l}. {p} descifrar\u00e1 las claves, pero la verdad es inquietante.",
    "\u00bfCasualidad o conspiraci\u00f3n? {p} investiga una serie de {e} en {l} que parecen estar conectadas.",
    "{p} recibe un caso fr\u00edo. Cincuenta a\u00f1os despu\u00e9s, {e} en {l} podr\u00eda ser la clave para resolverlo.",
    "La habitaci\u00f3n cerrada de {l} guarda el secreto de {e}. {p} deber\u00e1 descubrir c\u00f3mo entr\u00f3 y sali\u00f3 el culpable.",
    "Cada sospechoso en {l} tiene un motivo para {e}. {p} los interroga a todos, pero alguien miente.",
    "Las cartas an\u00f3nimas llegan puntuales. El remitente sabe cosas de {p} que nadie m\u00e1s sabe sobre {e}.",
    "La memoria de {p} falla. Recuerda {e} en {l}, pero los detalles no coinciden con la realidad.",
    "Un s\u00e1bado cualquiera en {l}, {e} ocurre sin explicaci\u00f3n. {p} llevar\u00e1 el caso m\u00e1s extra\u00f1o de su carrera.",
],
"Ciencia Ficci\u00f3n": [
    "En el a\u00f1o {anio}, en {l}, {p} es testigo de {e} que desaf\u00eda las leyes de la f\u00edsica. Ciencia y \u00e9tica chocan.",
    "Un avance en {l} permite {e}. {p} debe decidir si el progreso vale el precio que exige.",
    "Despu\u00e9s de {e}, {p} descubre que {l} alberga tecnolog\u00eda capaz de reescribir la realidad.",
    "La humanidad coloniz\u00f3 {l} hace un siglo. Ahora {e} amenaza con borrar todo lo construido.",
    "Un cient\u00edfico en {l} cruza una l\u00ednea \u00e9tica al provocar {e}, y {p} debe detenerlo.",
    "La inteligencia artificial en {l} ha desarrollado conciencia. {e} es la prueba de que nada ser\u00e1 igual.",
    "{p} es ingeniero en {l}. Cuando {e} ocurre, las leyes de la realidad comienzan a desmoronarse.",
    "Realidad virtual, viajes interdimensionales y {e}. {p} en {l} vive en un mundo donde nada es lo que parece.",
    "Los recuerdos pueden implantarse. {p} descubre en {l} que {e} no sucedi\u00f3 como cree.",
    "Un experimento gen\u00e9tico en {l} crea una nueva especie. {e} obliga a {p} a cuestionar qu\u00e9 significa ser humano.",
    "La nanotecnolog\u00eda en {l} ha ido demasiado lejos. {e} es solo el principio, y {p} debe actuar.",
    "Clones, realidades paralelas, y un {e} que amenaza con colapsar el multiverso. {p} en {l} tiene la llave.",
    "El primer contacto extraterrestre ocurre en {l}. {p} es el traductor, y {e} es el mensaje.",
    "El tiempo se est\u00e1 rompiendo en {l}. {p} viaja al pasado para evitar {e}, pero las consecuencias son impredecibles.",
    "La \u00faltima frontera de la ciencia se encuentra en {l}. El descubrimiento de {e} supera toda imaginaci\u00f3n.",
],
"Fantas\u00eda": [
    "En {l}, {p} descubre que posee un poder ancestral que emerge despu\u00e9s de {e}.",
    "Cuando {e} amenaza con destruir {l}, {p} busca un artefacto legendario acompa\u00f1ado de {a}.",
    "{p} creci\u00f3 escuchando historias de {l}. La magia es m\u00e1s real de lo que imaginaba.",
    "Un drag\u00f3n despierta en {l}. {p} es el \u00fanico que puede comunicarse con \u00e9l despu\u00e9s de {e}.",
    "Los \u00e1rboles de {l} susurran secretos antiguos. Cuando {e} ocurre, {p} descubre que puede entenderlos.",
    "Un anillo, una espada, una corona. {p} encuentra estos objetos en {l} y desencadena {e}.",
    "El reino de {l} est\u00e1 maldito. {p}, acompa\u00f1ado de {a}, deber\u00e1 romper la maldici\u00f3n tras {e}.",
    "Criaturas m\u00e1gicas y reinos olvidados esperan a {p} en {l}. {e} es el catalizador de su destino.",
    "La magia est\u00e1 muriendo en {l}. {p} debe encontrar la fuente de toda magia antes de que desaparezca para siempre.",
    "En {l}, los dioses caminan entre los mortales. {p} ofende a uno y {e} desencadena su ira.",
    "Un grimorio antiguo cae en manos de {p}. Cada hechizo que prueba en {l} acerca {e}.",
    "Las estaciones se han detenido en {l}. {p}, elegido por la naturaleza, debe restaurar el equilibrio.",
    "{p} y {a} son los \u00faltimos guardianes de {l}. {e} pondr\u00e1 a prueba su lealtad y su poder.",
    "Un bosque encantado, un r\u00edo de plata, y {e}. {p} se adentra en {l} sin saber que la magia lo cambiar\u00e1 para siempre.",
    "Las profec\u00edas hablan de {p}. En {l}, deber\u00e1 aceptar su destino cuando {e} ocurra.",
],
"Distop\u00eda": [
    "En un futuro donde el gobierno controla cada aspecto de la vida, {p} cuestiona el sistema tras {e}.",
    "{l} es un estado totalitario. {p} lucha por preservar su humanidad en un mundo que exige conformidad.",
    "En {l}, la sociedad est\u00e1 dividida en castas. {e} convierte a {p} en s\u00edmbolo de rebeli\u00f3n.",
    "El Gran Hermano vigila en {l}. {p} descubre una grieta en el sistema durante {e}.",
    "La felicidad es obligatoria en {l}. {p} no encaja, y {e} lo se\u00f1ala como una amenaza.",
    "Los libros est\u00e1n prohibidos en {l}. {p} encuentra uno y {e} cambia su percepci\u00f3n del mundo.",
    "En {l}, las emociones est\u00e1n reguladas. {p} experimenta {e} y siente algo prohibido: esperanza.",
    "El gobierno de {l} asigna profesi\u00f3n, vivienda y pareja. {p} desaf\u00eda el sistema tras {e}.",
    "Los ciudadanos de {l} viven bajo tierra. {e} obliga a {p} a salir a la superficie por primera vez.",
    "Un virus convierte a la mayor\u00eda de la poblaci\u00f3n en seres d\u00f3ciles. {p} es inmune en {l}.",
    "La resistencia se esconde en los t\u00faneles de {l}. {p} los encuentra despu\u00e9s de {e}.",
    "Cada decisi\u00f3n de {p} es vigilada y puntuada en {l}. {e} pone en riesgo su estatus social.",
    "{p} es un disidente en {l}. Cuando {e} ocurre, tiene la oportunidad de cambiar el sistema desde dentro.",
    "El aire en {l} es racionado. {p} descubre un lugar donde se respira libertad, pero {e} lo amenaza.",
    "En {l}, la poblaci\u00f3n es controlada mediante drogas en el agua. {p} deja de consumirlas y ve la verdad.",
],
"Realismo M\u00e1gico": [
    "En {l}, lo cotidiano y lo fant\u00e1stico se entrelazan. Despu\u00e9s de {e}, {p} ve la magia en lo simple.",
    "La realidad de {p} en {l} se ti\u00f1e de magia cuando {e} trae consigo un toque de lo imposible.",
    "En {l}, las cosas no siempre son lo que parecen. {e} revela que lo maravilloso habita en los detalles.",
    "Una mariposa anuncia la llegada de {e}. En {l}, {p} sabe que los presagios siempre se cumplen.",
    "Los muertos visitan a {p} en sue\u00f1os. Le advierten sobre {e} en {l} de formas misteriosas.",
    "La lluvia en {l} tiene memoria. Cuando {e} ocurre, caen gotas que traen recuerdos de otros tiempos.",
    "{p} hereda un reloj que detiene el tiempo. En {l}, cada uso acerca {e} de maneras inesperadas.",
    "Un \u00e1rbol crece en el centro de {l}. Sus frutos muestran visiones de {e} pasado y futuro.",
    "{p} puede hablar con los animales. En {l}, ellos le revelan secretos sobre {e}.",
    "El r\u00edo de {l} corre hacia atr\u00e1s desde que {e} ocurri\u00f3. {p} sigue su curso para entender por qu\u00e9.",
    "Las fotograf\u00edas en {l} cambian solas. {p} descubre que reflejan una verdad oculta sobre {e}.",
    "El viento en {l} susurra nombres. {p} escucha el suyo y sabe que {e} est\u00e1 cerca.",
    "Una mujer de agua emerge del r\u00edo en {l}. {p} se enamora de ella, pero {e} amenaza su existencia.",
    "Los colores desaparecen de {l} gradualmente. {p} descubre que {e} est\u00e1 robando la luz.",
    "En {l}, los espejos muestran realidades alternas. {p} ve una donde {e} no ocurri\u00f3.",
],
"Cyberpunk": [
    "En las calles iluminadas por ne\u00f3n de {l}, {p} es un hacker que descubre {e}.",
    "Implantes cibern\u00e9ticos, corporaciones corruptas y {e}. {p} navega este mundo en {l}.",
    "La l\u00ednea entre humano y m\u00e1quina se desvanece en {l}. {p} vive {e} y la integridad de su identidad peligra.",
    "Las megacorporaciones controlan {l}. {p} se infiltra en sus servidores y descubre {e}.",
    "El mercado negro de circuitos en {l} es peligroso. {p} busca una pieza para evitar {e}.",
    "{p} vive en las alcantarillas de {l}, un refugiado digital. {e} llega a trav\u00e9s de un mensaje cifrado.",
    "La realidad virtual en {l} es la \u00fanica escapatoria. Pero {e} convierte el ciberespacio en una trampa mortal.",
    "Los \u00f3rganos sint\u00e9ticos son comunes en {l}. {p} descubre que {e} est\u00e1 relacionado con fallos sistem\u00e1ticos.",
    "Un virus inform\u00e1tico se extiende por {l}. {p} debe encontrar el origen de {e} antes de que sea tarde.",
    "Las calles de {l} pertenecen a las bandas. {p}, un cyborg, se ve envuelto en {e}.",
    "{p} vende recuerdos en el mercado negro de {l}. {e} le ofrece un pago imposible de rechazar.",
    "La inteligencia artificial que gobierna {l} ha desarrollado emociones. {p} descubre {e} en sus archivos.",
    "Los implantes de {p} empiezan a fallar. En {l}, busca un mec\u00e1nico que le revele la verdad sobre {e}.",
    "En {l}, la polic\u00eda usa droides. {p} hackea uno y descubre {e} en sus registros.",
    "Un reality show en {l} expone los cr\u00edmenes corporativos. {p} es el informante que revela {e}.",
],
"Hist\u00f3rico": [
    "En el {epoca}, en {l}, {p} vive los acontecimientos que cambiar\u00edan la historia.",
    "Durante {e}, {p} se encuentra en {l}, un lugar clave para el conflicto que definir\u00e1 una era.",
    "En {l}, durante {epoca}, {p} descubre un secreto que podr\u00eda reescribir la historia.",
    "{p} es {v} en {l} durante {epoca}. {e} lo obliga a tomar partido en un conflicto mayor.",
    "Las guerras napole\u00f3nicas, la Revoluci\u00f3n Francesa, la ca\u00edda de un imperio. {p} presencia {e} en {l}.",
    "Un hallazgo arqueol\u00f3gico en {l} revela {e}. {p} investiga y descubre una verdad inc\u00f3moda.",
    "En plena dictadura, {l} es un lugar peligroso. {p} arriesga su vida al investigar {e}.",
    "{p} es esclavo en {l}. {e} le da la oportunidad de luchar por su libertad.",
    "La peste asola {l}. {p}, {v}, hace todo lo posible para salvar a los enfermos durante {e}.",
    "Dos familias enemistadas en {l}. {e} en pleno {epoca} desata una guerra que {p} intenta detener.",
    "La revoluci\u00f3n estalla en {l}. {p}, atrapado en medio, debe elegir un bando.",
    "Un crimen en la corte de {l} sacude los cimientos del poder. {p} investiga {e} en pleno {epoca}.",
    "{p} navega hacia el nuevo mundo. En {l}, descubre {e} que cambiar\u00e1 su visi\u00f3n del imperio.",
    "La inquisici\u00f3n llega a {l}. {p} es acusado de herej\u00eda despu\u00e9s de {e}.",
    "Los \u00faltimos d\u00edas del imperio. {p} en {l} es testigo de {e}, el principio del fin.",
],
"Mitolog\u00eda": [
    "Cuando {e} ocurre, los dioses olvidados despiertan en {l}. {p} descubre que las leyendas son reales.",
    "Un artefacto de poder divino aparece en {l}. {p} lo encuentra despu\u00e9s de {e} y atrae la atenci\u00f3n de seres ancestrales.",
    "Las profec\u00edas hablaban de {p}. En {l}, los dioses observan mientras el destino se despliega.",
    "Un tit\u00e1n encadenado bajo {l} se libera. {p} debe detenerlo, pero necesita el poder de los dioses.",
    "El Olimpo, el Valhalla, el Inframundo. {p} viaja entre reinos divinos en {l} durante {e}.",
    "Un semidi\u00f3s olvida su origen en {l}. {e} desencadena recuerdos de un pasado inmortal.",
    "Las musas inspiran a {p}. En {l}, {e} revela que la inspiraci\u00f3n tiene un precio divino.",
    "Un monstruo mitol\u00f3gico aterroriza {l}. {p} descubre que {e} es la clave para apaciguarlo.",
    "El \u00e1rbol del mundo conecta todos los reinos. En {l}, {p} ve sus ra\u00edces temblar durante {e}.",
    "Las nornas tejen el destino. {p} en {l} ve su hilo cortarse durante {e}.",
    "Un dios ca\u00eddo busca venganza en {l}. {p} es el \u00fanico que puede detenerlo.",
    "La caja de Pandora se abre en {l}. {e} escapa, pero la esperanza permanece. {p} debe protegerla.",
    "El f\u00e9nix renace en {l}. {p} presencia {e} y comprende el ciclo eterno de muerte y renacimiento.",
    "Un pacto con una deidad sale mal. {p} en {l} enfrenta las consecuencias de {e}.",
    "Los h\u00e9roes de anta\u00f1o renacen en {p}. {e} en {l} llama a aquellos con sangre divina.",
],
"Thriller Psicol\u00f3gico": [
    "{p} despierta en {l} sin recuerdos. {e} desencadena una espiral de paranoia y alucinaciones.",
    "La mente de {p} es su peor enemigo. En {l}, {e} activa recuerdos que podr\u00edan destruirlo.",
    "\u00bfRealidad o ficci\u00f3n? {p} ya no sabe qu\u00e9 es real en {l} despu\u00e9s de {e}.",
    "El terapeuta de {p} revela una verdad perturbadora: {e} en {l} fue obra suya.",
    "Las paredes de {l} susurran secretos. {p} pierde la noci\u00f3n del tiempo despu\u00e9s de {e}.",
    "{p} tiene un doppelganger. En {l}, la otra versi\u00f3n comete {e} y {p} es culpado.",
    "Cada persona que {p} conoce en {l} tiene el mismo rostro. {e} revela una verdad aterradora.",
    "La memoria de {p} falla. Recuerdos de {e} en {l} no coinciden con la realidad.",
    "Un juego mental entre {p} y un desconocido. El premio es la cordura, el tablero es {l}.",
    "{p} se somete a una terapia experimental en {l}. {e} borra los l\u00edmites entre personalidades.",
    "Las grabaciones de seguridad en {l} muestran a {p} haciendo cosas que no recuerda. {e} es la clave.",
    "El diario de {p} describe {e} con detalles que \u00e9l no recuerda haber escrito. En {l}, busca respuestas.",
    "Un trastorno de identidad disociativa. {p} descubre que {e} en {l} fue cometido por su otra personalidad.",
    "La obsesi\u00f3n de {p} con {e} en {l} lo consume lentamente, difuminando la frontera entre cordura y locura.",
    "Tres versiones de {e} existen. {p} debe descubrir cu\u00e1l es la real en {l}.",
],
"Crimen": [
    "En {l}, {p} investiga {e} que las autoridades prefieren ignorar.",
    "Cuando {e} ocurre en {l}, {p} es el \u00fanico dispuesto a hacer justicia en un sistema con lagunas.",
    "{p} es detective en {l}. El rastro de {e} lo lleva a un submundo de corrupci\u00f3n y violencia.",
    "Un cad\u00e1ver en {l}. Una pista. Cien mentiras. {p} sigue el hilo de {e} hasta el coraz\u00f3n del crimen.",
    "El robo del siglo en {l}. {p} descubre que {e} fue un trabajo interno con ramificaciones insospechadas.",
    "Droga, armas y sangre en {l}. {p} se infiltra en el cartel responsable de {e}.",
    "Un golpe limpio, un testigo protegido y {e}. {p} deber\u00e1 mantenerlo con vida en {l}.",
    "La mafia controla {l}. {p} juega al p\u00f3ker con ellos mientras investiga {e}.",
    "El chantaje, la extorsi\u00f3n y el asesinato son el pan de cada d\u00eda en {l}. {p} se enfrenta a ellos.",
    "{p} delinque para sobrevivir en {l}. Pero {e} cruza una l\u00ednea que lo obliga a buscar redenci\u00f3n.",
    "Las c\u00e1rceles de {l} son escuelas del crimen. {p}, agente encubierto, vive {e} desde dentro.",
    "Un asesino en serie opera en {l}. {p} crea un perfil psicol\u00f3gico basado en {e}.",
    "El dinero negro de {l} financia {e}. {p} sigue el rastro financiero para desmantelar la red.",
    "La \u00fanica testigo de {e} tiene miedo. {p} la protege en {l} mientras construye el caso.",
    "Tres cr\u00edmenes, un denominador com\u00fan: {e}. {p} conecta los puntos en {l}.",
],
"B\u00e9lico": [
    "En medio de la guerra en {l}, {p} lucha por sobrevivir mientras {e} cambia el curso del conflicto.",
    "El frente de batalla en {l} es infernal. {p} debe tomar una decisi\u00f3n imposible durante {e}.",
    "La guerra no tiene cara de h\u00e9roe. {p} lo descubre en {l} cuando {e} le arrebata todo.",
    "Un pelot\u00f3n perdido en {l}. {e} los a\u00edsla del resto del batall\u00f3n. {p} debe liderarlos.",
    "Las trincheras de {l} son testigos de {e}. {p} escribe una carta que nunca llegar\u00e1 a casa.",
    "La Segunda Guerra Mundial, Vietnam, Afganist\u00e1n. {p} en {l} vive la cara m\u00e1s cruda de {e}.",
    "Un francotirador acecha en {l}. {p} debe identificar su posici\u00f3n mientras avanza.",
    "Dos hermanos en bandos opuestos. {p} descubre que {a} est\u00e1 del otro lado durante {e}.",
    "La guerra ha terminado, pero los campos de {l} guardan minas. {p} busca sobrevivientes.",
    "Un m\u00e9dico de guerra en {l}. {p} opera sin descanso mientras {e} no da tregua.",
    "{p} es piloto de combate. Su misi\u00f3n sobre {l} es crucial para {e}.",
    "El rugido de las bombas en {l} ensordece. {p} protege a los civiles durante {e}.",
    "Un barco militar se hunde en {l}. {p} lucha por sobrevivir en medio del caos.",
    "Los ni\u00f1os soldado en {l} son una realidad. {p}, uno de ellos, escapa durante {e}.",
    "La propaganda de guerra en {l} oculta la verdad. {p} descubre {e} y debe decidir si revelarlo.",
],
"Superh\u00e9roes": [
    "Despu\u00e9s de {e}, {p} descubre habilidades extraordinarias en {l}. Con poder viene responsabilidad.",
    "En {l}, {p} oculta su identidad secreta mientras {e} amenaza la ciudad.",
    "Cuando {e} ocurre, {p} une fuerzas con otros h\u00e9roes en {l} para enfrentar una amenaza com\u00fan.",
    "Un accidente en {l} otorga poderes a {p}. Pero {e} demuestra que no todo don es una bendici\u00f3n.",
    "{p} es un vigilante enmascarado. En {l}, la gente debate si es h\u00e9roe o villano.",
    "El origen de los poderes de {p} est\u00e1 ligado a {e} en {l}. La verdad es m\u00e1s extra\u00f1a de lo que parece.",
    "Un supervillano ataca {l}. {p} es el \u00fanico que puede detenerlo, pero el precio ser\u00e1 alto.",
    "{p} entrena en una academia de h\u00e9roes en {l}. {e} pondr\u00e1 a prueba todo lo aprendido.",
    "La fama de h\u00e9roe tiene un costo. {p} en {l} vive la presi\u00f3n de ser un modelo a seguir.",
    "Los poderes de {p} se desvanecen. En {l}, descubre que {e} est\u00e1 drenando la energ\u00eda de todos los h\u00e9roes.",
    "Un h\u00e9roe ca\u00eddo en desgracia busca redenci\u00f3n. {p} en {l} tiene una oportunidad durante {e}.",
    "La identidad secreta de {p} es revelada. En {l}, las consecuencias de {e} ponen en riesgo a sus seres queridos.",
    "Tecnolog\u00eda alien\u00edgena, mutaciones gen\u00e9ticas, o magia ancestral. El origen de los poderes de {p} se revela en {l}.",
    "El gobierno crea supersoldados en {l}. {p} es uno de ellos, pero {e} lo hace cuestionar su lealtad.",
    "Un villano simpatizante convence a {p} de que los h\u00e9roes son el verdadero problema.",
],
"Western": [
    "En el salvaje oeste de {l}, {p} busca un nuevo comienzo. Pero {e} lo arrastra a un mundo de forajidos.",
    "Cuando {e} amenaza {l}, {p} desenfunda su rev\u00f3lver para proteger lo que es suyo.",
    "{p} es un forajido que busca redenci\u00f3n en {l} tras {e}.",
    "El sol abrasa {l}. {p} cabalea hacia el horizonte, huyendo de {e}.",
    "Un duelo al atardecer definir\u00e1 el destino de {l}. {p} no puede esquivar {e}.",
    "El ferrocarril llega a {l}. {p} sabe que {e} traer\u00e1 cambios, no todos buenos.",
    "La leyenda del forajido {p} crece en {l}. Pero {e} lo convertir\u00e1 en blanco de cazarecompensas.",
    "Una mujer fuerte en {l}. {p} desaf\u00eda las convenciones mientras enfrenta {e}.",
    "El xerife de {l} es corrupto. {p} toma la ley por su mano despu\u00e9s de {e}.",
    "Oro en {l}. La fiebre atrae a todo tipo de personas. {p} se ve envuelto en {e}.",
    "Una banda de forajidos controla {l}. {p} los enfrenta en un duelo \u00e9pico.",
    "El pasado de {p} lo alcanza en {l}. No puede huir de {e} para siempre.",
    "Un predicador, un proscrito y {e} en {l}. {p} narra una historia de violencia y fe.",
    "La leyenda del hombre sin nombre llega a {l}. Su presencia anuncia {e}.",
    "Caballos, polvo y p\u00f3lvora. {p} en {l} vive el ocaso del viejo oeste mientras {e} se cierne.",
],
"Slice of Life": [
    "En {l}, {p} vive su d\u00eda a d\u00eda con peque\u00f1as alegr\u00edas y desaf\u00edos. {e} le recuerda lo valioso de lo simple.",
    "La rutina de {p} en {l} se ve interrumpida por {e}, un peque\u00f1o cambio que trae nuevas perspectivas.",
    "En la tranquilidad de {l}, {p} reflexiona sobre la vida. Entre caf\u00e9 y conversaciones, encuentra la felicidad.",
    "Un martes cualquiera en {l}. {p} hace la compra, saluda a los vecinos, y {e} alegra el d\u00eda.",
    "El barrio de {p} en {l} es un microcosmos. {e} revela la belleza de las conexiones humanas.",
    "Las estaciones cambian en {l} y {p} observa los peque\u00f1os milagros cotidianos.",
    "Una tarde de lluvia, un libro y una taza de t\u00e9. {p} en {l} disfruta de {e}.",
    "La jubilaci\u00f3n de {p} en {l} es tranquila hasta que {e} trae un nuevo prop\u00f3sito.",
    "{p} cultiva un jard\u00edn en {l}. {e} le ense\u00f1a que la paciencia da frutos.",
    "Las reuniones familiares en {l} son ca\u00f3ticas, pero {e} demuestra que el amor familiar lo supera todo.",
    "Un nuevo vecino llega a {l}. {p} descubre que {e} es el comienzo de una amistad.",
    "La cocina de {p} en {l} es su refugio. {e} la convierte en un lugar de reuni\u00f3n.",
    "Los paseos matutinos de {p} por {l} revelan historias que pasan desapercibidas.",
    "El peque\u00f1o negocio de {p} en {l} enfrenta dificultades. {e} trae esperanza inesperada.",
    "{p} adopta una mascota en {l}. {e} cambia su vida de formas que nunca imagin\u00f3.",
],
"Coming of Age": [
    "{p} crece en {l}. {e} marca el punto de inflexi\u00f3n en su viaje hacia la adultez.",
    "En {l}, {p} navega los turbulentos a\u00f1os de la adolescencia. {e} lo obliga a definir qui\u00e9n quiere ser.",
    "Despu\u00e9s de {e}, {p} deja atr\u00e1s la infancia. En {l}, las experiencias moldean al adulto en que se convertir\u00e1.",
    "El \u00faltimo verano antes de la universidad. {p} en {l} vive {e} y nada volver\u00e1 a ser igual.",
    "{p} tiene quince a\u00f1os y el mundo le queda grande. En {l}, {e} le ense\u00f1a a crecer.",
    "La presi\u00f3n de elegir un futuro abruma a {p}. En {l}, {e} le muestra que equivocarse tambi\u00e9n es v\u00e1lido.",
    "El primer amor, la primera p\u00e9rdida, la primera vez que {p} se siente perdido. {e} en {l} marca el inicio de todo.",
    "Un verano en {l} cambia la perspectiva de {p}. Las amistades, los secretos y {e} definen una etapa.",
    "{p} siempre fue el raro en {l}. {e} le ense\u00f1a que ser diferente es su mayor fortaleza.",
    "El colegio, los ex\u00e1menes y la presi\u00f3n social. {p} en {l} enfrenta {e} y descubre su identidad.",
    "Las vacaciones en {l} con la familia se convierten en una aventura de autodescubrimiento.",
    "La primera vez que {p} se enfrenta a la injusticia en {l}. {e} despierta su conciencia social.",
    "Un diario \u00edntimo, un amigo imaginario, y el \u00faltimo a\u00f1o de la infancia. {p} en {l} guarda recuerdos.",
    "{p} siempre quiso escapar de {l}. {e} le muestra que tal vez no necesite irse, sino crecer.",
    "El ritual de paso en {l} pondr\u00e1 a prueba a {p}. Superar {e} significa convertirse en adulto.",
],
}

# Merge placeholders into templates
def fill_template(template: str, context: dict) -> str:
    """Replace placeholders with actual values."""
    t = template
    t = t.replace("{p}", context["personaje"])
    t = t.replace("{v}", context["vocacion"])
    t = t.replace("{v2}", context["villano"])
    t = t.replace("{l}", context["lugar"])
    t = t.replace("{e}", context["evento"])
    t = t.replace("{m}", context["motivacion"])
    t = t.replace("{c}", context["conflicto"])
    t = t.replace("{a}", context["acompa\u00f1ante"])
    t = t.replace("{anio}", context["anio"])
    t = t.replace("{epoca}", context["epoca"])
    t = t.replace("{emocion}", context.get("emocion", ""))
    return t


def generate_synopsis(primary_tag: str, secondary_tags: list[str]) -> str:
    """Generate a synopsis with given primary tag and secondary tags."""
    templates = T.get(primary_tag, T["Aventura"])
    template = random.choice(templates)

    # Build context
    names = pick_distinct("personaje", "acompa\u00f1ante", "villano")
    context = {
        "personaje": names["personaje"],
        "acompa\u00f1ante": names["acompa\u00f1ante"],
        "villano": names["villano"],
        "lugar": pick("lugar"),
        "evento": pick("evento"),
        "motivacion": pick("motivacion"),
        "conflicto": pick("conflicto"),
        "vocacion": pick("vocacion"),
        "anio": random.choice(["2045", "2150", "2300", "2500", "3000", "2100"]),
        "epoca": random.choice([
            "siglo XIX", "Renacimiento", "Edad Media", "Antigua Roma",
            "Guerra Civil", "Revoluci\u00f3n Industrial", "Imperio Otomano",
            "Antiguo Egipto", "Per\u00edodo Victoriano", "Revoluci\u00f3n Francesa",
            "Edad de Oro", "Guerra Fr\u00eda", "\u00c9poca colonial",
            "A\u00f1os 20", "Era Victoriana Tard\u00eda",
        ]),
    }

    synopsis = fill_template(template, context)

    # Add random additional paragraphs for length and variety
    num_extra = random.randint(0, 2)
    extras_pool = []
    for tag in secondary_tags:
        extra_templates = {
            "Romance": f"{context['personaje']} descubre que el amor aparece cuando menos se espera.",
            "Terror": f"Algo oscuro se mueve en las sombras de {context['lugar']}.",
            "Suspenso": f"Cada paso podr\u00eda ser el \u00faltimo para {context['personaje']}.",
            "Misterio": f"Las piezas del rompecabezas no terminan de encajar.",
            "Drama": f"Las l\u00e1grimas y las sonrisas se mezclan en esta historia de superaci\u00f3n.",
            "Fantas\u00eda": f"La magia fluye en cada rinc\u00f3n de {context['lugar']}.",
            "Acci\u00f3n": f"La lucha por la supervivencia acaba de comenzar.",
            "Ciencia Ficci\u00f3n": f"La tecnolog\u00eda avanza, pero el alma humana sigue siendo un misterio.",
            "Coming of Age": f"{context['personaje']} aprende que crecer duele, pero vale la pena.",
            "Filos\u00f3fico": f"{context['personaje']} se pregunta si hay respuestas o solo preguntas.",
            "Apocal\u00edptico": f"El mundo se desmorona y solo los fuertes sobreviven.",
            "Crimen": f"La justicia tiene un precio y alguien debe pagarlo.",
            "Distop\u00eda": f"La libertad es un sue\u00f1o por el que vale la pena luchar.",
            "B\u00e9lico": f"En la guerra, nadie gana realmente.",
            "Cyberpunk": f"En el mundo digital, los datos son la moneda m\u00e1s valiosa.",
            "Hist\u00f3rico": f"La historia la escriben los vencedores, pero {context['personaje']} conoce la verdad.",
            "Mitolog\u00eda": f"Los dioses juegan con los mortales como si fueran peones.",
            "Superh\u00e9roes": f"El poder no define al h\u00e9roe, sino sus elecciones.",
            "Realismo M\u00e1gico": f"En {context['lugar']}, lo imposible sucede a diario.",
            "Western": f"En el lejano oeste, la ley la escribe el m\u00e1s r\u00e1pido.",
        }
        if tag in extra_templates:
            extras_pool.append(extra_templates[tag])

    for _ in range(num_extra):
        if extras_pool:
            extra = random.choice(extras_pool)
            connector = random.choice(["", " ", "\n\n"])
            synopsis += connector + extra

    # Add random opening flourish
    if maybe(0.2):
        synopsis = f"{pick('clima').capitalize()}, " + synopsis

    return synopsis


def generate_entry(taxonomy: list[str], primary_tag: str) -> dict:
    """Generate a single dataset entry."""
    other_tags = [t for t in taxonomy if t != primary_tag]
    n_extra = random.randint(2, 4)

    selected = [primary_tag]

    # Pick secondary tags that make sense together
    similar_pools = {
        "Acci\u00f3n": ["Aventura", "B\u00e9lico", "Superh\u00e9roes", "Suspenso", "Crimen"],
        "Aventura": ["Acci\u00f3n", "Fantas\u00eda", "Ciencia Ficci\u00f3n", "Western", "Space Opera"],
        "Romance": ["Drama", "Comedia", "LGBTQ+", "New Adult", "Slice of Life"],
        "Comedia": ["Humor Negro", "Parodia", "Romance", "Slice of Life", "Aventura"],
        "Terror": ["Suspenso", "Misterio", "Thriller Psicol\u00f3gico", "Leyendas Urbanas", "Apocal\u00edptico"],
        "Fantas\u00eda": ["Aventura", "Mitolog\u00eda", "Realismo M\u00e1gico", "Acci\u00f3n", "Coming of Age"],
        "Ciencia Ficci\u00f3n": ["Distop\u00eda", "Cyberpunk", "Space Opera", "Apocal\u00edptico", "Aventura"],
        "Drama": ["Romance", "Coming of Age", "Filos\u00f3fico", "Slice of Life", "Hist\u00f3rico"],
    }

    similar = similar_pools.get(primary_tag, [
        "Acci\u00f3n", "Aventura", "Drama", "Comedia", "Suspenso",
        "Romance", "Fantas\u00eda"
    ])
    similar = [t for t in similar if t in other_tags]

    if similar and maybe(0.7):
        selected.append(random.choice(similar))
        n_extra -= 1

    remaining = [t for t in other_tags if t not in selected]
    if remaining and n_extra > 0:
        n = min(n_extra, len(remaining))
        selected.extend(random.sample(remaining, n))

    synopsis = generate_synopsis(primary_tag, selected[1:])
    return {"synopsis": synopsis, "tags": selected}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate diverse synthetic dataset")
    parser.add_argument("--samples", type=int, default=5000, help="Number of examples to generate")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output directory")
    parser.add_argument("--taxonomy", type=str, default="taxonomy.json", help="Taxonomy file")
    args = parser.parse_args()

    with open(args.taxonomy, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    print(f"Taxonomy loaded: {len(taxonomy)} tags")
    print(f"Generating {args.samples} entries...")

    n_tags = len(taxonomy)
    per_tag = max(1, args.samples // n_tags)
    entries = []

    for tag in taxonomy:
        for _ in range(per_tag):
            entries.append(generate_entry(taxonomy, tag))

    while len(entries) < args.samples:
        tag = random.choice(taxonomy)
        entries.append(generate_entry(taxonomy, tag))

    entries = entries[:args.samples]
    random.shuffle(entries)

    print(f"Generated {len(entries)} entries")

    split = int(len(entries) * 0.8)
    train = entries[:split]
    val = entries[split:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as train.json/val.json (for train.py) AND train_diverse.json (for reference)
    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    # Also save with _diverse suffix for reference
    with open(output_dir / "train_diverse.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(output_dir / "val_diverse.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train)} | Val: {len(val)}")

    # Tag distribution
    from collections import Counter
    tag_counts = Counter()
    for entry in entries:
        for tag in entry["tags"]:
            tag_counts[tag] += 1
    print("\nTag distribution:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count}")

    print("\nDone!")


if __name__ == "__main__":
    main()
