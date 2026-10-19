// ImpactGuide · the names of the bench's parts in words, for the name shown beside a part picked in 3D.
//
// The scene's objects have names made for the scene (pendulum_mass_washer_l2, a1_ball_stud...): each is matched
// against this list, in order, the first that fits giving its name in the page's language (English when the
// language has none); a name the list does not know is shown as it is, its underscores as spaces.
'use strict';

const PARTS = {
  // [pattern of the scene's name, en, it, fr, es, de, zh]
  LIST: [
    [/^a1_(arm)/, 'Holder A arm', 'Braccio del supporto A', 'Bras du support A', 'Brazo del soporte A', 'Arm des Halters A', '支架 A 臂'],
    [/^a1_ball_stud/, 'Holder A ball stud', 'Perno a sfera del supporto A', 'Rotule du support A', 'Perno de bola del soporte A', 'Kugelbolzen des Halters A', '支架 A 球头螺柱'],
    [/^a1_cap/, 'Holder A cap', 'Cappuccio del supporto A', 'Capuchon du support A', 'Tapa del soporte A', 'Kappe des Halters A', '支架 A 盖'],
    [/^a1_cup/, 'Holder A ball cup', 'Coppa del supporto A', 'Coupelle du support A', 'Copa del soporte A', 'Kugelpfanne des Halters A', '支架 A 球碗'],
    [/^a1_discs/, 'Holder A friction discs', 'Dischi di attrito del supporto A', 'Disques de friction du support A', 'Discos de fricción del soporte A', 'Reibscheiben des Halters A', '支架 A 摩擦片'],
    [/^a1_/, 'Holder A screws', 'Viti del supporto A', 'Vis du support A', 'Tornillos del soporte A', 'Schrauben des Halters A', '支架 A 螺钉'],
    [/^b1_balls/, 'Holder B balls', 'Sfere del supporto B', 'Billes du support B', 'Bolas del soporte B', 'Kugeln des Halters B', '支架 B 钢球'],
    [/^b1_base/, 'Holder B base', 'Base del supporto B', 'Base du support B', 'Base del soporte B', 'Basis des Halters B', '支架 B 底座'],
    [/^b1_insert/, 'Holder B insert', 'Inserto del supporto B', 'Insert du support B', 'Inserto del soporte B', 'Einsatz des Halters B', '支架 B 嵌件'],
    [/^b1_pins/, 'Holder B pins', 'Perni del supporto B', 'Goupilles du support B', 'Pasadores del soporte B', 'Stifte des Halters B', '支架 B 销'],
    [/^b1_plate/, 'Holder B plate', 'Piastra del supporto B', 'Plaque du support B', 'Placa del soporte B', 'Platte des Halters B', '支架 B 板'],
    [/^b1_spring/, 'Holder B spring', 'Molla del supporto B', 'Ressort du support B', 'Muelle del soporte B', 'Feder des Halters B', '支架 B 弹簧'],
    [/^b1_/, 'Holder B screws', 'Viti del supporto B', 'Vis du support B', 'Tornillos del soporte B', 'Schrauben des Halters B', '支架 B 螺钉'],
    [/^arm_(bearings|thrust_bearing)/, 'Arm bearings', 'Cuscinetti del braccio', 'Roulements du bras', 'Rodamientos del brazo', 'Lager des Arms', '臂轴承'],
    [/^arm_(bogie|wheels)/, 'Arm carriage', 'Carrello del braccio', 'Chariot du bras', 'Carro del brazo', 'Wagen des Arms', '臂滑车'],
    [/^arm_boom_phone/, 'Phone boom', 'Asta del telefono', 'Perche du téléphone', 'Pértiga del teléfono', 'Handyausleger', '手机悬臂'],
    [/^arm_(collar|hub|pivot_shaft)/, 'Arm pivot', 'Perno del braccio', 'Pivot du bras', 'Pivote del brazo', 'Drehpunkt des Arms', '臂枢轴'],
    [/^arm_(end_cap|post_cap)/, 'End cap', 'Tappo', 'Embout', 'Tapón', 'Endkappe', '端盖'],
    [/^arm_plunger/, 'Arm index plunger', 'Pistoncino di posizione del braccio', "Doigt d'indexage du bras", 'Émbolo indexador del brazo', 'Indexbolzen des Arms', '臂分度销'],
    [/^arm_post/, 'Arm post', 'Colonna del braccio', 'Colonne du bras', 'Columna del brazo', 'Säule des Arms', '臂立柱'],
    [/^arm_profile/, 'Arm profile', 'Profilo del braccio', 'Profilé du bras', 'Perfil del brazo', 'Profil des Arms', '臂型材'],
    [/^(arm|light)_slot_covers/, 'Slot covers', 'Copricava', 'Couvre-rainures', 'Cubreranuras', 'Nutabdeckungen', '槽盖条'],
    [/^arm_track_scale/, 'Arm scale', 'Scala del braccio', 'Échelle du bras', 'Escala del brazo', 'Skala des Arms', '臂刻度'],
    [/^arm_track/, 'Arm track', 'Binario del braccio', 'Rail du bras', 'Carril del brazo', 'Laufbahn des Arms', '臂轨道'],
    [/^base_board/, 'Base board', 'Piano di base', 'Plateau de base', 'Tablero base', 'Grundplatte', '底板'],
    [/^base_foot/, 'Feet', 'Piedini', 'Pieds', 'Patas', 'Füße', '底脚'],
    [/^cable_phone/, 'Phone cable', 'Cavo del telefono', 'Câble du téléphone', 'Cable del teléfono', 'Handykabel', '手机线缆'],
    [/^cable_(power|hub_pwr)/, 'Power cable', 'Cavo di alimentazione', "Câble d'alimentation", 'Cable de alimentación', 'Stromkabel', '电源线'],
    [/^cable_(usb|hub_up|led\d_usb)/, 'USB cable', 'Cavo USB', 'Câble USB', 'Cable USB', 'USB-Kabel', 'USB 线'],
    [/^cable_led/, 'Light cable', 'Cavo della luce', "Câble de l'éclairage", 'Cable de la luz', 'Leuchtenkabel', '灯光线缆'],
    [/^cables_(arm_row_straps|clips|tags)/, 'Cable clips', 'Fermacavi', 'Attaches de câbles', 'Sujetacables', 'Kabelclips', '线夹'],
    [/^(cables_connectors|cables_plugs|phone_usb_plug|scanner_plugs)/, 'Connectors', 'Connettori', 'Connecteurs', 'Conectores', 'Stecker', '接头'],
    [/^connector_skull/, 'Skull connector', 'Raccordo del cranio', 'Raccord du crâne', 'Conector del cráneo', 'Schädelverbinder', '颅骨连接件'],
    [/^drape/, 'Drape', 'Telo', 'Champ', 'Paño', 'Abdecktuch', '手术巾'],
    [/^face_magnet/, 'Face magnets', 'Magneti del volto', 'Aimants du visage', 'Imanes de la cara', 'Gesichtsmagnete', '面部磁铁'],
    [/^gt_frame_mandible/, 'Mandible reference frame', 'Telaio di riferimento della mandibola', 'Cadre de référence de la mandibule', 'Marco de referencia de la mandíbula', 'Referenzrahmen des Unterkiefers', '下颌参考架'],
    [/^gt_sphere_mandible/, 'Mandible reference sphere', 'Sfera di riferimento della mandibola', 'Sphère de référence de la mandibule', 'Esfera de referencia de la mandíbula', 'Referenzkugel des Unterkiefers', '下颌参考球'],
    [/^gt_sphere_marker/, 'Marker reference sphere', 'Sfera di riferimento del marker', 'Sphère de référence du marqueur', 'Esfera de referencia del marcador', 'Referenzkugel des Markers', '标记参考球'],
    [/^led/, 'Light controller', 'Centralina della luce', "Contrôleur d'éclairage", 'Controlador de la luz', 'Lichtsteuerung', '灯光控制器'],
    [/^light_mount/, 'Light mount', 'Attacco della luce', "Fixation de l'éclairage", 'Montura de la luz', 'Leuchtenhalter', '灯座'],
    [/^light_panel/, 'Light panel', 'Pannello luminoso', 'Panneau lumineux', 'Panel de luz', 'Leuchtpanel', '灯板'],
    [/^light_/, 'Light stand', 'Supporto delle luci', "Support d'éclairage", 'Soporte de las luces', 'Leuchtenstativ', '灯架'],
    [/^mandible/, 'Mandible', 'Mandibola', 'Mandibule', 'Mandíbula', 'Unterkiefer', '下颌骨'],
    [/^marker_charuco/, 'ChArUco marker', 'Marker ChArUco', 'Marqueur ChArUco', 'Marcador ChArUco', 'ChArUco-Marker', 'ChArUco 标记'],
    [/^marker_holder/, 'Marker holder', 'Supporto del marker', 'Support du marqueur', 'Soporte del marcador', 'Markerhalter', '标记支架'],
    [/^marker_plate/, 'Marker plate', 'Piastra del marker', 'Plaque du marqueur', 'Placa del marcador', 'Markerplatte', '标记板'],
    [/^marker_strike_cap/, 'Strike cap', "Cappuccio d'urto", "Capuchon d'impact", 'Tapa de impacto', 'Aufschlagkappe', '撞击帽'],
    [/^pendulum_mass_washer/, 'Mass washers', 'Rondelle della massa', 'Rondelles de la masse', 'Arandelas de la masa', 'Massescheiben', '配重垫圈'],
    [/^pendulum_mass/, 'Pendulum mass', 'Massa del pendolo', 'Masse du pendule', 'Masa del péndulo', 'Pendelmasse', '摆锤'],
    [/^pendulum_(arm|beam|strut)/, 'Pendulum arm', 'Braccio del pendolo', 'Bras du pendule', 'Brazo del péndulo', 'Pendelarm', '摆臂'],
    [/^pendulum_(axle|bearing|hub|head)/, 'Pendulum axle', 'Asse del pendolo', 'Axe du pendule', 'Eje del péndulo', 'Pendelachse', '摆轴'],
    [/^pendulum_(column|frame)/, 'Pendulum column', 'Colonna del pendolo', 'Colonne du pendule', 'Columna del péndulo', 'Pendelsäule', '摆柱'],
    [/^pendulum_(guide_scale|protractor)/, 'Pendulum protractor', 'Goniometro del pendolo', 'Rapporteur du pendule', 'Transportador del péndulo', 'Pendelwinkelmesser', '摆角度尺'],
    [/^pendulum_guide/, 'Pendulum guide', 'Guida del pendolo', 'Guide du pendule', 'Guía del péndulo', 'Pendelführung', '摆导轨'],
    [/^pendulum_(index|latch)/, 'Pendulum index', 'Fermo del pendolo', 'Indexage du pendule', 'Indexador del péndulo', 'Pendelraste', '摆分度器'],
    [/^pendulum_release/, 'Release', 'Sgancio', 'Déclencheur', 'Disparador', 'Auslöser', '释放机构'],
    [/^pendulum_(thrust_bearing|turntable)/, 'Pendulum turntable', 'Piatto girevole del pendolo', 'Plateau tournant du pendule', 'Plato giratorio del péndulo', 'Pendeldrehteller', '摆转台'],
    [/^pendulum_tip/, 'Pendulum tip', 'Punta del pendolo', 'Pointe du pendule', 'Punta del péndulo', 'Pendelspitze', '摆头'],
    [/^pendulum_truck/, 'Pendulum carriage', 'Carrello del pendolo', 'Chariot du pendule', 'Carro del péndulo', 'Pendelwagen', '摆滑车'],
    [/^pendulum/, 'Pendulum', 'Pendolo', 'Pendule', 'Péndulo', 'Pendel', '摆'],
    [/^phantom_post/, 'Phantom post', 'Colonna del fantoccio', 'Colonne du fantôme', 'Columna del maniquí', 'Phantomsäule', '模型立柱'],
    [/^phone_cradle/, 'Phone cradle', 'Culla del telefono', 'Support du téléphone', 'Soporte del teléfono', 'Handyhalter', '手机夹'],
    [/^phone_tilt/, 'Phone tilt', 'Inclinazione del telefono', 'Inclinaison du téléphone', 'Inclinación del teléfono', 'Handyneigung', '手机倾角调节'],
    [/^scanner_ball_head/, 'Scanner ball head', 'Testa a sfera dello scanner', 'Rotule du scanner', 'Rótula del escáner', 'Kugelkopf des Scanners', '扫描仪球头'],
    [/^scanner_body/, 'Scanner', 'Scanner', 'Scanner', 'Escáner', 'Scanner', '扫描仪'],
    [/^scanner_hand/, 'Scanner clamp', 'Morsa dello scanner', 'Pince du scanner', 'Pinza del escáner', 'Scannerklemme', '扫描仪夹具'],
    [/^scanner_lock/, 'Scanner lock knob', 'Pomello di blocco dello scanner', 'Bouton de blocage du scanner', 'Pomo de bloqueo del escáner', 'Klemmknopf des Scanners', '扫描仪锁紧旋钮'],
    [/^scanner_pedestal/, 'Scanner pedestal', 'Piedistallo dello scanner', 'Socle du scanner', 'Pedestal del escáner', 'Scannersockel', '扫描仪底座'],
    [/^scanner_rail/, 'Scanner rail', 'Guida dello scanner', 'Rail du scanner', 'Carril del escáner', 'Scannerschiene', '扫描仪导轨'],
    [/^screw/, 'Screw', 'Vite', 'Vis', 'Tornillo', 'Schraube', '螺钉'],
    [/^washer/, 'Washer', 'Rondella', 'Rondelle', 'Arandela', 'Unterlegscheibe', '垫圈'],
    [/^silicone_face/, 'Silicone face', 'Volto in silicone', 'Visage en silicone', 'Cara de silicona', 'Silikongesicht', '硅胶面部'],
    [/^skull/, 'Skull', 'Cranio', 'Crâne', 'Cráneo', 'Schädel', '颅骨'],
    [/^smartphone/, 'Phone', 'Telefono', 'Téléphone', 'Teléfono', 'Handy', '手机'],
    [/^splint/, 'Splint', 'Splint', 'Gouttière', 'Férula', 'Schiene', '咬合板'],
    [/^surgical_cap/, 'Surgical cap', 'Cuffia chirurgica', 'Calot chirurgical', 'Gorro quirúrgico', 'OP-Haube', '手术帽'],
    [/^usb_hub/, 'USB hub', 'Hub USB', 'Hub USB', 'Hub USB', 'USB-Hub', 'USB 集线器'],
    // the printed plate (ui/models/charuco_plate.glb)
    [/^white$/, 'Plate', 'Piastra', 'Plaque', 'Placa', 'Platte', '底板'],
    [/^black$/, 'ChArUco print', 'Stampa ChArUco', 'Impression ChArUco', 'Impresión ChArUco', 'ChArUco-Druck', 'ChArUco 印刷图案'],
  ],
  LANGS: ['en', 'it', 'fr', 'es', 'de', 'zh'],

  // the scene's name of a part in words, in the page's language
  name: function (tech) {
    const t = String(tech || '');
    const row = PARTS.LIST.find(function (r) { return r[0].test(t); });
    if (!row) return t.replace(/[._]+/g, ' ').replace(/\s\d+$/, '').trim();
    const k = PARTS.LANGS.indexOf(typeof LANG === 'string' ? LANG : 'en');
    return row[1 + Math.max(0, k)] || row[1];
  },

  // whether the list knows a name (the tests: every part of the bench has its words)
  known: function (tech) { return PARTS.LIST.some(function (r) { return r[0].test(String(tech || '')); }); },
};
