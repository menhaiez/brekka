# -*- coding: utf-8 -*-
"""
Brekka · подготовка моделей барельефа к 3-осевой фрезеровке.

Два инструмента:

1. Heightfield-пересборка (главный режим)
   Модель сэмплируется сверху в высотную карту, наклон которой ограничен
   заданным углом. Гарантирует отсутствие крутых стенок и поднутрений.
   Низ можно удлинить "юбкой" для последующего скульпта.

2. Анализ для скульпта
   Раскраска модели прямо в Sculpt Mode:
   - красный   — круче заданного угла / поднутрение;
   - оранжевый — конусная фреза не проходит по углу;
   - маджента — щель уже диаметра фрезы: она туда не пролезет;
   - синий     — жертвенная юбка.

Установка: Edit > Preferences > Add-ons > Install from Disk... > этот файл.
Панель: 3D Viewport > N > вкладка "Brekka".
"""

bl_info = {
    "name": "Brekka",
    "author": "Vadim Yunevich",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Brekka",
    "description": "Heightfield-фаска и анализ поднутрений для 3-осевой фрезеровки",
    "category": "Mesh",
}

import math
import time

import bmesh
import bpy
import numpy as np
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
)
from bpy.types import Operator, Panel
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

# Область пересборки: длина диагонали, к которой приводятся радиусы (сетка ~0.1 мм).
REFERENCE_DIAG = 90.0


# ===========================================================================
#  1. Heightfield-пересборка
# ===========================================================================

def _find_back_plane(bm, total_area, scale):
    """Найти плоскую спину модели независимо от ориентации.

    Возвращает (rotation, normal, area) или None.
    *rotation* переводит нормаль спины в -Z: спина оказывается снизу.
    """
    bins = {}
    for f in bm.faces:
        n = f.normal
        if n.length_squared < 0.25:
            continue
        key = (round(n.x * 8.0), round(n.y * 8.0), round(n.z * 8.0))
        bins[key] = bins.get(key, 0.0) + f.calc_area()
    if not bins:
        return None

    need = 0.02 * total_area
    tol = max(1e-4, 0.15 * scale)

    for key, barea in sorted(bins.items(), key=lambda kv: -kv[1])[:4]:
        if barea < need:
            continue
        kn = Vector((key[0] / 8.0, key[1] / 8.0, key[2] / 8.0))
        if kn.length_squared < 1e-8:
            continue
        kn.normalize()

        acc = Vector((0.0, 0.0, 0.0))
        aacc = 0.0
        for f in bm.faces:
            if f.normal.dot(kn) > 0.985:
                a = f.calc_area()
                acc += f.normal * a
                aacc += a
        if aacc < need:
            continue
        n = acc.normalized()

        # спина — крайняя сторона: почти все вершины лежат по одну сторону
        for sign in (1.0, -1.0):
            nn = n * sign
            dmax = max(v.co.dot(nn) for v in bm.verts)
            area = 0.0
            for f in bm.faces:
                if f.normal.dot(nn) > 0.99 and \
                        all(abs(v.co.dot(nn) - dmax) <= tol for v in f.verts):
                    area += f.calc_area()
            if area >= need:
                q = nn.to_track_quat('-Z', 'Y')
                return q.inverted().to_matrix(), nn, area
    return None


def _sample_heightfield(bm, res, zmin, zmax, x0g, x1g, y0g, y1g):
    """Лучи сверху вниз: высота поверхности на регулярной сетке."""
    nx = int((x1g - x0g) / res) + 5
    ny = int((y1g - y0g) / res) + 5
    if nx * ny > 25_000_000:
        raise RuntimeError(
            "Слишком мелкий шаг сетки: %d x %d ячеек. Увеличьте шаг." % (nx, ny)
        )
    x0 = x0g - 2.0 * res
    y0 = y0g - 2.0 * res
    xs = x0 + np.arange(nx) * res
    ys = y0 + np.arange(ny) * res

    bvh = BVHTree.FromBMesh(bm)
    field = np.zeros((nx, ny), np.float32)
    down = Vector((0.0, 0.0, -1.0))
    for i in range(nx):
        for j in range(ny):
            hit = bvh.ray_cast(Vector((xs[i], ys[j], zmax + 5.0)), down)
            if hit[0] is not None:
                h = hit[0].z - zmin
                if h > 1e-5:
                    field[i, j] = h
    return field, xs, ys, nx, ny


def _limit_slope(field, res, draft_deg, max_iter=30):
    """Ограничить наклон: u = min_q [f(q) + |p-q| / tan(draft)]."""
    c = 1.0 / math.tan(math.radians(draft_deg))
    u = field.copy()
    w1 = c * res
    w2 = w1 * math.sqrt(2.0)
    steps = [(-1, 0, w1), (1, 0, w1), (0, -1, w1), (0, 1, w1),
             (-1, -1, w2), (-1, 1, w2), (1, -1, w2), (1, 1, w2)]
    nx, ny = field.shape
    for _ in range(max_iter):
        padded = np.pad(u, 1, constant_values=0)
        cand = u.copy()
        for di, dj, w in steps:
            cand = np.minimum(
                cand, padded[1 + di:1 + di + nx, 1 + dj:1 + dj + ny] + w
            )
        done = np.abs(cand - u).max() < 1e-4
        u = cand
        if done:
            break
    return u


def _build_field_mesh(field, u, xs, ys, nx, ny, zmin):
    """Сетка по высотной карте: лицевая поверхность + плоская спина."""
    inside = field > 0
    padded = np.pad(inside, 1, constant_values=False)
    active = np.zeros_like(inside)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            active |= padded[1 + di:1 + di + nx, 1 + dj:1 + dj + ny]

    cells = inside[:-1, :-1] | inside[1:, :-1] | inside[:-1, 1:] | inside[1:, 1:]
    ci, cj = np.nonzero(cells)
    if len(ci) == 0:
        raise RuntimeError("Не удалось построить высотную карту (пустая проекция)")

    front_index = -np.ones((nx, ny), np.int64)
    ii, jj = np.nonzero(active)
    n_front = len(ii)
    front_index[ii, jj] = np.arange(n_front)

    back_index = -np.ones((nx, ny), np.int64)
    bi, bj = np.nonzero(inside)
    n_back = len(bi)
    back_index[bi, bj] = np.arange(n_back)

    verts = np.empty((n_front + n_back, 3))
    verts[:n_front, 0] = xs[ii]
    verts[:n_front, 1] = ys[jj]
    verts[:n_front, 2] = zmin + u[ii, jj]
    verts[n_front:, 0] = xs[bi]
    verts[n_front:, 1] = ys[bj]
    verts[n_front:, 2] = zmin

    faces = []
    for a, b in zip(ci, cj):
        corners = ((a, b), (a + 1, b), (a + 1, b + 1), (a, b + 1))
        flags = [1 if inside[i, j] else 0 for (i, j) in corners]
        fv = [front_index[i, j] for (i, j) in corners]
        bv = [n_front + back_index[i, j] if inside[i, j] else front_index[i, j]
              for (i, j) in corners]

        # Если внутри только один угол, режем диагональю через него: иначе
        # автотриангуляция создаёт плоский треугольник, дублирующий спину.
        if sum(flags) == 1:
            k = flags.index(1)
            order = [(k + n) % 4 for n in range(4)]
        else:
            order = [0, 1, 2, 3]
        o0, o1, o2, o3 = order
        faces.append((fv[o0], fv[o1], fv[o2]))
        faces.append((fv[o0], fv[o2], fv[o3]))
        faces.append((bv[o0], bv[o2], bv[o1]))
        faces.append((bv[o0], bv[o3], bv[o2]))

    return verts, faces, active


def _smooth_border(mesh, active, inside, xs, ys, zmin,
                   iterations, radius, strength):
    """Сгладить полосу у кромки, чтобы скрыть ступеньки сетки.

    Рим (z == zmin) закреплён: спина остаётся плоской.
    """
    if iterations <= 0 or radius <= 0:
        return
    border = np.nonzero(active & ~inside)
    if not len(border[0]):
        return
    bxy = np.column_stack((xs[border[0]], ys[border[1]]))
    kd = KDTree(len(bxy))
    for k in range(len(bxy)):
        kd.insert(Vector((bxy[k, 0], bxy[k, 1], 0.0)), k)
    kd.balance()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    selected = []
    for v in bm.verts:
        if v.co.z <= zmin + 1e-4:
            continue
        _, _, dist = kd.find(Vector((v.co.x, v.co.y, 0.0)))
        if dist <= radius:
            selected.append(v)

    for _ in range(int(iterations)):
        moved = {}
        for v in selected:
            if not v.link_edges:
                continue
            avg = Vector((0.0, 0.0, 0.0))
            for e in v.link_edges:
                avg += e.other_vert(v).co
            moved[v] = v.co.lerp(avg / len(v.link_edges), strength)
        for v, co in moved.items():
            v.co = co

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# ===========================================================================
#  Юбка: создание и восстановление после ремеша
# ===========================================================================

SKIRT_GROUP = "draft_skirt"     # имя группы, оставлено для старых .blend
SKIRT_EXT = "hf_base_extend"    # сколько опущено дно


def _grow_skirt(obj, depth):
    """Наростить юбку: опустить спинку модели на *depth* вниз.

    Отдельный шаг после пересборки (и после ремеша). Спинка = плоские
    нижние грани на минимальной отметке. Возвращает (сколько опущено, z_спины).
    """
    me = obj.data
    if not depth or depth <= 0 or len(me.vertices) == 0:
        return None
    if not len(me.polygons):
        return None

    mode = obj.mode
    if mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    try:
        coords = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", coords)
        zmin = float(coords.reshape(-1, 3)[:, 2].min())

        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        caps = [f for f in bm.faces
                if f.normal.z < -0.9
                and all(abs(v.co.z - zmin) < 1e-4 for v in f.verts)]
        if not caps:
            bm.free()
            return None
        result = bmesh.ops.extrude_face_region(bm, geom=caps)
        new_verts = [g for g in result.get("geom", [])
                     if isinstance(g, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, vec=Vector((0.0, 0.0, -float(depth))),
                            verts=new_verts)
        bmesh.ops.delete(bm, geom=caps, context='FACES')
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.to_mesh(me)
        bm.free()
        me.update()
    finally:
        if mode != 'OBJECT':
            bpy.ops.object.mode_set(mode=mode)

    # вершины новой спинки: всё, что на глубине ниже прежнего дна
    coords = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", coords)
    z = coords.reshape(-1, 3)[:, 2]
    idx = np.nonzero(z < zmin - 1e-4)[0]

    old_group = obj.vertex_groups.get(SKIRT_GROUP)
    if old_group is not None:
        obj.vertex_groups.remove(old_group)
    group = obj.vertex_groups.new(name=SKIRT_GROUP)
    if len(idx):
        group.add([int(i) for i in idx], 1.0, 'REPLACE')

    obj[SKIRT_EXT] = float(depth)
    return len(idx), zmin


def _skirt_band_verts(obj, depth):
    """Вершины в полосе юбки: от низа модели до низа + depth."""
    me = obj.data
    n = len(me.vertices)
    if n == 0:
        return np.zeros(0, np.int64), 0.0
    coords = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", coords)
    z = coords.reshape(-1, 3)[:, 2]
    zmin = float(z.min())
    return np.nonzero(z < zmin + float(depth))[0], zmin


def _mesh_stats(mesh, axis=None):
    """Максимальный наклон, площадь крутых граней, неманифолдные рёбра.

    *axis* — нормаль лицевой стороны в мировых координатах (None => Z).
    """
    if axis is None:
        axis = Vector((0.0, 0.0, 1.0))
    axis = axis.normalized()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    betas = []
    areas = []
    for f in bm.faces:
        areas.append(f.calc_area())
        cos = abs(f.normal.dot(axis))
        betas.append(math.degrees(math.acos(min(1.0, cos))))
    nonmanifold = sum(1 for e in bm.edges if len(e.link_faces) != 2)
    bm.free()
    areas = np.asarray(areas)
    betas = np.asarray(betas)
    return {
        'triangles': len(areas),
        'total_area': float(areas.sum()),
        'max_beta': float(betas.max()) if len(betas) else 0.0,
        'steep_area': float(areas[betas > 89.5].sum()),
        'nonmanifold': nonmanifold,
    }


def heightfield_draft(obj, wall_angle_deg=91.5, resolution=0.10,
                      keep_original=True, margin_deg=0.7, smooth_iters=20,
                      smooth_radius=0.8, smooth_strength=0.5):
    """Пересобрать рельеф как heightfield с ограниченным наклоном.

    Возвращает (object, stats). Ориентация объекта значения не имеет:
    плоскость спины находится автоматически, после обработки модель
    возвращается в исходную ориентацию.
    """
    if obj.type != 'MESH':
        raise RuntimeError("Объект не является мешем")
    me = obj.data
    if len(me.vertices) == 0 or len(me.polygons) == 0:
        raise RuntimeError("Меш пустой")

    draft_deg = max(0.1, wall_angle_deg - 90.0 + margin_deg)
    res = max(0.01, float(resolution))

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    coords = [v.co for v in bm.verts]
    lo = Vector((min(c.x for c in coords), min(c.y for c in coords),
                 min(c.z for c in coords)))
    hi = Vector((max(c.x for c in coords), max(c.y for c in coords),
                 max(c.z for c in coords)))
    diag = (hi - lo).length
    total_area = sum(f.calc_area() for f in bm.faces)
    scale = diag / REFERENCE_DIAG

    found = _find_back_plane(bm, total_area, scale)
    if found is None:
        bm.free()
        raise RuntimeError(
            "Не найден большой плоский участок (спинка). Нужна модель с "
            "более-менее плоской спиной (допускается волна до ~0.15)."
        )
    rotation = found[0]
    for v in bm.verts:
        v.co = rotation @ v.co
    bm.normal_update()

    coords = [v.co for v in bm.verts]
    zmin = min(c.z for c in coords)
    zmax = max(c.z for c in coords)
    x0g = min(c.x for c in coords)
    x1g = max(c.x for c in coords)
    y0g = min(c.y for c in coords)
    y1g = max(c.y for c in coords)

    field, xs, ys, nx, ny = _sample_heightfield(bm, res, zmin, zmax, x0g, x1g, y0g, y1g)
    bm.free()

    u = _limit_slope(field, res, draft_deg)
    verts, faces, active = _build_field_mesh(field, u, xs, ys, nx, ny, zmin)

    mesh = bpy.data.meshes.new("heightfield")
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.validate()
    mesh.update()

    if smooth_iters > 0 and smooth_radius > 0:
        _smooth_border(mesh, active, field > 0, xs, ys, zmin,
                       smooth_iters, smooth_radius * res / 0.12,
                       smooth_strength)

    # ось переда в исходной ориентации (нормаль спины, развёрнутая на перед)
    front_axis = rotation.transposed() @ Vector((0.0, 0.0, 1.0))
    stats = _mesh_stats(mesh, front_axis)

    # обратно в исходную ориентацию
    back = rotation.transposed()
    for v in mesh.vertices:
        v.co = back @ v.co
    mesh.update()

    stats.update({'resolution': res, 'draft_deg': draft_deg})

    if keep_original:
        new_obj = bpy.data.objects.new(obj.name + "_HF", mesh)
        new_obj.matrix_world = obj.matrix_world.copy()
        collection = obj.users_collection[0] if obj.users_collection else \
            bpy.context.scene.collection
        collection.objects.link(new_obj)
        return new_obj, stats

    old = obj.data
    obj.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return obj, stats


# ===========================================================================
#  2. Анализ для скульпта
# ===========================================================================

COL_OK = (0.55, 0.60, 0.55, 1.0)
COL_RED = (0.90, 0.02, 0.02, 1.0)
COL_ORANGE = (1.00, 0.55, 0.00, 1.0)      # V: не проходит по углу
COL_DARK_ORANGE = (0.80, 0.28, 0.00, 1.0)  # сверху перекрыто
COL_MAGENTA = (0.85, 0.00, 0.75, 1.0)      # фреза не пролезает
COL_SKIRT = (0.22, 0.30, 0.42, 1.0)
ANALYSIS_ATTR = "draft_analysis"   # имя атрибута, оставлено для старых .blend

# для сравнения цветов из массива (float32) без ошибок округления
_PALETTE = np.array([COL_OK, COL_DARK_ORANGE, COL_ORANGE, COL_RED,
                     COL_MAGENTA, COL_SKIRT], np.float32)

# 8 горизонтальных направлений для проверки ширины щели
_DIRS8 = ((1.0, 0.0), (0.7071, 0.7071), (0.0, 1.0), (-0.7071, 0.7071),
          (-1.0, 0.0), (-0.7071, -0.7071), (0.0, -1.0), (0.7071, -0.7071))
_DIRS4 = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))

# (значение, иконка, название) — кнопки выбора фрезы
CUTTERS = (
    ('BALL', 'SPHERE', 'brekka.cutter_ball'),
    ('FLAT', 'MESH_CYLINDER', 'brekka.cutter_flat'),
    ('TAPER', 'CONE', 'brekka.cutter_taper'),
)

_analysis_state = {}
_analysis_visible = False
_zoom_cursor = 0
def _scene_settings():
    scene = getattr(bpy.context, "scene", None)
    if scene is None or not hasattr(scene, "brekka"):
        return None
    return scene.brekka


def _view_is_vertex():
    """Виден ли сейчас Attribute-цвет в 3D-окне."""
    screen = getattr(bpy.context, "screen", None)
    if not screen:
        return True
    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for space in area.spaces:
            if space.type == 'VIEW_3D' and space.shading.color_type != 'VERTEX':
                return False
    return True


def _stop_analysis_tick():
    if bpy.app.timers.is_registered(_analysis_tick):
        bpy.app.timers.unregister(_analysis_tick)


def _set_view_color(mode):
    screen = getattr(bpy.context, "screen", None)
    if not screen:
        return
    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                try:
                    space.shading.color_type = mode
                except Exception:
                    pass


def _clear_other_analysis(active_obj):
    """Убрать раскраску у всех мешей, кроме активного."""
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or obj is active_obj or obj.data is active_obj.data:
            continue
        attr = obj.data.color_attributes.get(ANALYSIS_ATTR)
        if attr is not None:
            try:
                obj.data.color_attributes.remove(attr)
            except Exception:
                pass


def _slot_width(center, bvh, radius, dirs=_DIRS8):
    """Ширина щели: расстояния до стенок в противоположных направлениях."""
    reach = 2.0 * radius
    o = center + Vector((0.0, 0.0, max(1e-3, radius * 0.35)))
    dist = []
    for dx, dy in dirs:
        hit = bvh.ray_cast(o, Vector((dx, dy, 0.0)), reach)
        dist.append(hit[3] if hit[0] is not None else 1.0e9)
    half = len(dirs) // 2
    return min(dist[i] + dist[i + half] for i in range(half))


def _short_color(center, normal, bvh, radius, kind, min_nz, zmin,
                 skirt_z, min_radius, dirs=_DIRS8, check_draft=True,
                 check_tool=True, taper_angle=4.88):
    """Цвет грани: красный (уклон), оранжевый (фреза по углу), маджента
    (не пролезет), юбка. Остальное — норма."""
    nz = normal.z
    if center.z < skirt_z - 1e-4:
        return COL_SKIRT
    if nz < -0.5 and center.z < zmin + 1e-3:
        return COL_OK
    if check_draft and nz < min_nz:
        return COL_RED
    if radius <= 0.0 or not check_tool:
        return COL_OK
    if kind == 'TAPER':
        # конус не пройдёт стенку круче своего уклона
        if nz < math.sin(math.radians(max(0.1, taper_angle))) - 1e-4:
            return COL_ORANGE
    if nz > 0.85 and min_radius > 0.0:
        if _slot_width(center, bvh, min_radius, dirs) < 2.0 * min_radius - 1e-9:
            return COL_MAGENTA
    return COL_OK


def _read_geometry(me):
    n = len(me.polygons)
    centers = np.empty(n * 3, np.float32)
    me.polygons.foreach_get("center", centers)
    normals = np.empty(n * 3, np.float32)
    me.polygons.foreach_get("normal", normals)
    loop_totals = np.empty(n, np.int32)
    me.polygons.foreach_get("loop_total", loop_totals)
    loop_verts = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_verts)
    verts = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", verts)
    return {
        'centers': centers.reshape(n, 3),
        'normals': normals.reshape(n, 3),
        'loop_totals': loop_totals,
        'loop_verts': loop_verts,
        'verts': verts.reshape(-1, 3),
    }


def _analysis_full(obj, radius, min_nz, context, kind='BALL',
                   min_radius=0.25, check_draft=True, check_tool=True,
                   taper_angle=4.88):
    """Полный расчёт: (красные, оранжевые, тёмно-оранжевые, всего, маджента)."""
    me = obj.data
    n = len(me.polygons)
    if n > ANALYSIS_MAX_FACES:
        raise RuntimeError(
            "Слишком тяжёлая модель: %d граней (лимит %d). Упростите до ~1 млн."
            % (n, ANALYSIS_MAX_FACES))
    geo = _read_geometry(me)
    zmin = float(geo['verts'][:, 2].min())
    ext = float(obj.get(SKIRT_EXT, 0.0) or 0.0)
    skirt_z = (zmin + ext) if ext > 0.0 else zmin

    bvh = BVHTree.FromObject(obj, context.evaluated_depsgraph_get())
    colors = np.empty((n, 4), np.float32)
    colors[:] = COL_OK
    red = 0
    angle = 0
    phys = 0
    tight = 0
    wm = getattr(bpy.context, "window_manager", None)
    try:
        if wm is not None:
            wm.progress_begin(0, n)
    except Exception:
        wm = None
    try:
        for i in range(n):
            color = _short_color(Vector(geo['centers'][i]),
                                 Vector(geo['normals'][i]), bvh, radius, kind,
                                 min_nz, zmin, skirt_z, min_radius,
                                 _DIRS8, check_draft, check_tool, taper_angle)
            colors[i] = color
            if color is COL_RED:
                red += 1
            elif color is COL_ORANGE:
                angle += 1
            elif color is COL_DARK_ORANGE:
                phys += 1
            elif color is COL_MAGENTA:
                tight += 1
            if wm is not None and i % 50000 == 0:
                try:
                    wm.progress_update(i)
                except Exception:
                    wm = None
    finally:
        if wm is not None:
            try:
                wm.progress_end()
            except Exception:
                pass

    attr = me.color_attributes.get(ANALYSIS_ATTR)
    if attr is None or attr.domain != 'CORNER':
        if attr is not None:
            try:
                me.color_attributes.remove(attr)
            except Exception:
                pass
        # CORNER: у каждой грани свой плоский цвет — границы меток резкие
        attr = me.color_attributes.new(name=ANALYSIS_ATTR, type='BYTE_COLOR',
                                       domain='CORNER')
    loop_colors = np.repeat(colors, geo['loop_totals'], axis=0)
    attr.data.foreach_set("color", loop_colors.ravel())
    try:
        me.color_attributes.active_color = attr
    except Exception:
        pass
    me.update()
    _set_view_color('VERTEX')

    global _analysis_visible, _zoom_cursor
    _analysis_visible = True
    _zoom_cursor = 0

    # все грани каждой вершины (CSR): нужно, чтобы перекрашивать все
    # соседние грани, а не только одну
    counts = np.bincount(geo['loop_verts'], minlength=len(geo['verts']))
    vert_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    face_of_loop = np.repeat(np.arange(n, dtype=np.int64), geo['loop_totals'])
    vert_faces = face_of_loop[np.argsort(geo['loop_verts'], kind='stable')]

    _analysis_state.clear()
    _analysis_state.update({
        'obj': obj.name,
        'verts': geo['verts'],
        'v_offsets': vert_offsets,
        'v_faces': vert_faces,
        'face_colors': colors,
        'attr': attr,
        'radius': radius,
        'kind': kind,
        'min_nz': min_nz,
        'min_radius': min_radius,
        'check_draft': bool(check_draft),
        'check_tool': bool(check_tool),
        'taper_angle': float(taper_angle),
        'taper_nz': math.sin(math.radians(max(0.1, taper_angle))),
        'zmin': zmin,
        'skirt_z': skirt_z,
        # кэш буфера и отметки изменений для дешёвого тика
        'scratch': np.empty(len(geo['verts']) * 3, np.float32),
        'diff': np.empty(len(geo['verts']) * 3, bool),
        'changed': np.zeros(len(geo['verts']), bool),
        'n_verts': len(geo['verts']),
        'loop_total': geo['loop_totals'],
        'loop_verts': geo['loop_verts'],
    })
    return red, angle, phys, n, tight


def _mesh_face_normals(me):
    n = len(me.polygons)
    nor = np.empty(n * 3, np.float32)
    me.polygons.foreach_get("normal", nor)
    return nor.reshape(-1, 3)


def _mesh_face_areas(me):
    n = len(me.polygons)
    area = np.empty(n, np.float32)
    me.polygons.foreach_get("area", area)
    return area


def _laplacian_smooth(me, indices, factor, clamp=0.3):
    """Сглаживание Лапласа с ограничением шага.

    Шаг вершины не больше `clamp` от её самого короткого ребра — иначе на
    плотной сетке тонкие грани вытягиваются в иглы и выворачиваются
    (в Solid они чёрные). Возвращает число сдвинутых вершин.
    """
    nv = len(me.vertices)
    if not nv or not len(indices):
        return 0
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)

    ne = len(me.edges)
    if not ne:
        return 0
    ev = np.empty(ne * 2, np.int32)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)

    deg = np.zeros(nv, np.float32)
    acc = np.zeros((nv, 3), np.float64)
    np.add.at(deg, ev[:, 0], 1.0)
    np.add.at(deg, ev[:, 1], 1.0)
    np.add.at(acc, ev[:, 0], co[ev[:, 1]])
    np.add.at(acc, ev[:, 1], co[ev[:, 0]])

    idx = np.asarray(indices, np.int64)
    idx = idx[(idx >= 0) & (idx < nv)]
    if not len(idx):
        return 0
    d = deg[idx]
    avg = np.zeros((len(idx), 3), np.float64)
    nz = d > 0
    avg[nz] = acc[idx[nz]] / d[nz][:, None]
    avg[~nz] = co[idx[~nz]]

    delta = (avg - co[idx]) * factor
    # ограничение шага по короткому ребру
    elen = np.linalg.norm(co[ev[:, 0]] - co[ev[:, 1]], axis=1)
    short = np.full(nv, np.inf, np.float32)
    np.minimum.at(short, ev[:, 0], elen)
    np.minimum.at(short, ev[:, 1], elen)
    lim = short[idx] * clamp
    dl = np.linalg.norm(delta, axis=1)
    scale = np.minimum(1.0, lim / np.maximum(dl, 1e-9))
    co[idx] += (delta * scale[:, None]).astype(np.float32)
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    return int(len(idx))


AUTO_MAX_FACES = 2_500_000      # выше — живой пересчёт не включаем
PHYS_LIVE_MAX = 40000           # сколько помеченных граней пересчитывать на мазке
MARK_RECHECK_MAX = 1500         # ... но не больше этого за один тик (лучи дорогие)
ANALYSIS_MAX_FACES = 3_000_000  # выше — полный анализ слишком долгий


def _near_marked(me, marked_idx, moved_coords, pad):
    """Индексы помеченных граней в кубе вокруг изменённых вершин."""
    lo = moved_coords.min(axis=0) - pad
    hi = moved_coords.max(axis=0) + pad
    polys = me.polygons
    out = []
    for i in marked_idx.tolist():
        c = polys[i].center
        if lo[0] <= c.x <= hi[0] and lo[1] <= c.y <= hi[1] and \
                lo[2] <= c.z <= hi[2]:
            out.append(i)
    return out


SHORT_LIVE_MAX = 5000           # мелкий мазок — считаем всё, крупный — только цвет по нормали


def _tick_bvh(obj, max_age=0.5):
    """BVH по текущей геометрии, но не чаще раза в max_age секунд."""
    st = _analysis_state
    now = time.time()
    bvh = st.get('bvh')
    if bvh is not None and now - st.get('bvh_t', 0.0) < max_age:
        return bvh
    try:
        bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    except Exception:
        return None
    st['bvh'] = bvh
    st['bvh_t'] = now
    return bvh


def _analysis_tick():
    """Таймер: перекрасить только грани, задетые кистью.

    Красный/оранжевый — векторно по нормалям; маджента — лучами по BVH
    (BVH кэшируется на полсекунды, чтобы не строить его каждый тик).
    Возвращает паузу; None останавливает таймер.
    """
    t0 = time.perf_counter()
    st = _analysis_state
    settings = _scene_settings()
    if not st or settings is None:
        return None
    if not settings.analysis_auto or not _analysis_visible:
        return None
    wait = max(0.2, settings.analysis_update_ms / 1000.0)
    obj = bpy.data.objects.get(st['obj'])
    if not obj or obj.type != 'MESH':
        _analysis_state.clear()
        return None
    if obj.mode != 'SCULPT':
        return wait
    me = obj.data
    n_polys = len(me.polygons)
    if len(me.vertices) != st['n_verts'] or n_polys != len(st['face_colors']):
        _analysis_state.clear()
        return None
    if n_polys > AUTO_MAX_FACES:
        _analysis_state.clear()
        return None
    if n_polys > 1_500_000:
        wait = max(wait, 0.6)
    if not _view_is_vertex():
        return wait

    verts = st['scratch']
    me.vertices.foreach_get("co", verts)
    np.not_equal(verts, st['verts'].ravel(), out=st['diff'])
    changed = st['changed']
    np.any(st['diff'].reshape(-1, 3), axis=1, out=changed)
    moved = np.nonzero(changed)[0]
    if not len(moved):
        return wait

    offs = st['v_offsets']
    vfaces = st['v_faces']
    counts = offs[moved + 1] - offs[moved]
    total = int(counts.sum())
    if not total:
        st['verts'][:] = verts.reshape(-1, 3)
        return wait
    starts = np.repeat(offs[moved], counts)
    base = np.repeat(np.cumsum(counts) - counts, counts)
    faces = vfaces[starts + (np.arange(total) - base)].astype(np.int64)
    faces = np.unique(faces)
    faces = faces[(faces >= 0) & (faces < n_polys)]

    if len(faces) * 4 < n_polys:
        centers = np.empty((len(faces), 3), np.float32)
        normals = np.empty((len(faces), 3), np.float32)
        polys = me.polygons
        for k, f in enumerate(faces.tolist()):
            poly = polys[f]
            centers[k] = poly.center
            normals[k] = poly.normal
    else:
        fcenters = np.empty(n_polys * 3, np.float32)
        fnormals = np.empty(n_polys * 3, np.float32)
        me.polygons.foreach_get("center", fcenters)
        me.polygons.foreach_get("normal", fnormals)
        centers = fcenters.reshape(-1, 3)[faces]
        normals = fnormals.reshape(-1, 3)[faces]

    face_colors = st['face_colors']
    was_magenta = np.all(face_colors[faces] == _PALETTE[4], axis=1)
    changed_color = False

    # старые маджента-метки рядом с кистью (без BVH — только bbox)
    mag = np.nonzero(np.all(face_colors == _PALETTE[4], axis=1))[0]
    near = []
    if len(mag):
        mv = verts.reshape(-1, 3)[moved]
        ext = float(np.linalg.norm(mv.max(axis=0) - mv.min(axis=0)))
        near = _near_marked(me, mag, mv, max(ext * 0.12, 1e-4))
        near = near[:MARK_RECHECK_MAX]

    bvh = None
    if len(faces) <= SHORT_LIVE_MAX or near:
        bvh = _tick_bvh(obj)

    if bvh is not None and len(faces) <= SHORT_LIVE_MAX:
        for k in range(len(faces)):
            f = int(faces[k])
            c = _short_color(Vector(centers[k]), Vector(normals[k]), bvh,
                             st['radius'], st['kind'],
                             st['min_nz'], st['zmin'], st['skirt_z'],
                             st['min_radius'], _DIRS4,
                             st['check_draft'], st['check_tool'],
                             st['taper_angle'])
            if not np.array_equal(face_colors[f], c):
                face_colors[f] = c
                changed_color = True
    else:
        nz = normals[:, 2]
        skirt = centers[:, 2] < st['skirt_z'] - 1e-4
        down = nz < -0.5
        red = (nz < st['min_nz']) & ~skirt & ~down
        if not st['check_draft']:
            red = np.zeros(len(faces), bool)
        orange = np.zeros(len(faces), bool)
        if st['check_tool'] and st['kind'] == 'TAPER':
            orange = (nz < st['taper_nz']) & ~skirt & ~down & ~red
        face_colors[faces] = COL_OK
        if len(skirt):
            face_colors[faces[skirt]] = COL_SKIRT
        if len(red):
            face_colors[faces[red]] = COL_RED
        if len(orange):
            face_colors[faces[orange]] = COL_ORANGE
        keep = was_magenta & ~red & ~orange & ~skirt
        if not st['check_tool']:
            keep = np.zeros(len(faces), bool)
        if len(keep):
            face_colors[faces[keep]] = COL_MAGENTA
        changed_color = True

    if near and bvh is not None:
        polys = me.polygons
        for i in near:
            poly = polys[i]
            c = _short_color(Vector(poly.center), Vector(poly.normal), bvh,
                             st['radius'], st['kind'],
                             st['min_nz'], st['zmin'], st['skirt_z'],
                             st['min_radius'], _DIRS8,
                             st['check_draft'], st['check_tool'],
                             st['taper_angle'])
            if not np.array_equal(face_colors[i], c):
                face_colors[i] = c
                changed_color = True

    if changed_color:
        loop_colors = np.repeat(face_colors, st['loop_total'], axis=0)
        st['attr'].data.foreach_set("color", loop_colors.ravel())
        me.update()

    st['verts'][:] = verts.reshape(-1, 3)
    cost = time.perf_counter() - t0
    if cost > 0.4:
        wait = min(4.0, max(wait, cost * 1.6))
    return wait


@bpy.app.handlers.persistent
def _on_file_load(_dummy=None):
    """Новый файл: сбросить состояние и остановить таймер."""
    global _analysis_visible
    _analysis_visible = False
    _analysis_state.clear()
    _stop_analysis_tick()


# ===========================================================================
#  3. Настройки
# ===========================================================================

def _checks_toggled(self, context):
    """Галочки «Уклоны»/«Проход фрезы»: перекрасить сразу, если анализ виден."""
    obj = context.active_object
    if obj and obj.type == 'MESH' and _analysis_visible and _analysis_state:
        try:
            _recolor_analysis(obj, self, context)
        except Exception:
            pass


def _auto_toggled(self, context):
    """Флажок «Автообновление»: сразу включить или остановить таймер."""
    if not self.analysis_auto:
        _stop_analysis_tick()
        return
    obj = context.active_object
    if not obj or obj.type != 'MESH':
        return
    if len(obj.data.polygons) > AUTO_MAX_FACES:
        return
    if _analysis_state and _analysis_visible and \
            not bpy.app.timers.is_registered(_analysis_tick):
        bpy.app.timers.register(_analysis_tick, first_interval=0.3)


class BrekkaSettings(bpy.types.PropertyGroup):
    wall_angle: FloatProperty(
        name="Угол стены",
        description="Угол между лицевой плоскостью и стенкой (90 = вертикально)",
        default=91.5, min=90.1, max=95.0, step=10, precision=2,
    )
    hf_resolution: FloatProperty(
        name="Шаг сетки",
        description="Разрешение пересборки в единицах модели",
        default=0.12, min=0.04, max=0.40, precision=3,
    )
    hf_smooth_iters: IntProperty(
        name="Сглаживание кромки",
        description="Проходов сглаживания у нижней кромки (0 = выключено)",
        default=20, min=0, max=40,
    )
    hf_keep_original: BoolProperty(
        name="Оставить оригинал",
        description="Создать новый объект, исходный не трогать",
        default=True,
    )
    hf_base_extend: FloatProperty(
        name="Удлинить дно",
        description="Опустить дно на N вниз: жертвенная юбка для скульпта",
        default=0.5, min=0.0, max=10.0, precision=3,
    )
    d_ball: FloatProperty(
        name="Ø шара, мм", description="Диаметр шаровой фрезы",
        default=1.0, min=0.1, max=20.0, precision=2,
    )
    d_flat: FloatProperty(
        name="Ø плоской, мм", description="Диаметр плоской фрезы",
        default=1.0, min=0.1, max=20.0, precision=2,
    )
    d_taper: FloatProperty(
        name="Ø конусной, мм",
        description="Диаметр сферического носика конусной фрезы (R = Ø/2)",
        default=0.5, min=0.1, max=20.0, precision=2,
    )
    taper_angle: FloatProperty(
        name="Угол конуса, °",
        description="Угол наклона боковой кромки конусной фрезы (A на картинке)",
        default=4.88, min=0.5, max=45.0, precision=2,
    )

    def _get_diameter(self):
        return getattr(self, "d_" + self.cutter_type.lower())

    def _set_diameter(self, value):
        setattr(self, "d_" + self.cutter_type.lower(), value)

    analysis_diameter: FloatProperty(
        name="Диаметр фрезы, мм",
        description="Диаметр активной фрезы (у каждого типа свой)",
        get=_get_diameter, set=_set_diameter,
        min=0.1, max=20.0, precision=2,
    )
    cutter_type: EnumProperty(
        name="Фреза",
        description="Тип фрезы для анализа: шар, плоская или конусная",
        items=(
            ('BALL', "Шар", "Шаровая фреза"),
            ('FLAT', "Плоская", "Плоская фреза с острой кромкой"),
            ('TAPER', "Конусная", "Конусная сферическая: R0.25, уклон 4.88°"),
        ),
        default='BALL',
    )
    analysis_update_ms: IntProperty(
        name="Пауза, мс",
        description=("Пауза после мазка. Больше — легче скульпту, но подсветка "
                     "следует за кистью медленнее"),
        default=400, min=200, max=2000, step=50,
    )
    remesh_target: IntProperty(
        name="Цель, млн",
        description="Целевое число вершин после воксельного ремеша",
        default=5, min=1, max=20,
    )
    simplify_target: FloatProperty(
        name="До, млн",
        description="До какого числа вершин упростить модификатором Decimate",
        default=1.0, min=0.1, max=10.0, precision=1,
    )
    check_draft: BoolProperty(
        name="Уклоны",
        description="Красным подсветит грани, которым не хватает уклона: "
                    "они круче угла стены (91.5°)",
        default=True, update=_checks_toggled,
    )
    check_tool: BoolProperty(
        name="Проход фрезы",
        description="Оранжевым/маджентой — места, куда не пройдёт активная фреза",
        default=True, update=_checks_toggled,
    )
    analysis_auto: BoolProperty(
        name="Автообновление",
        description="Пересчитывать подсветку после каждого мазка кистью",
        default=True,
        update=_auto_toggled,
    )

    # --- свёрнутые блоки панели ---
    show_hf: BoolProperty(name="Пересборка", default=True)
    show_skirt: BoolProperty(name="Юбка", default=True)
    show_remesh: BoolProperty(name="Ремеш", default=False)
    show_analysis: BoolProperty(name="Анализ для скульпта", default=True)


def _settings(context):
    return context.scene.brekka


def _min_nz(settings):
    """Минимальная z-нормаль, при которой грань ещё не красная."""
    return math.sin(math.radians(max(0.1, settings.wall_angle - 90.0)))


def _mesh_target(op, context):
    """Активный меш или None (с сообщением об ошибке)."""
    obj = context.active_object
    if not obj or obj.type != "MESH":
        op.report({'ERROR'}, "Выделите меш-модель")
        return None
    return obj


# ===========================================================================
#  4. Операторы
# ===========================================================================

def _report_analysis(op, name, stats):
    op.report({'INFO'},
              "%s: %d треугольников | макс. наклон %.2f° (угол у лица %.2f°) | "
              "круче 89.5°: %.2f | неманифолдных рёбер %d" %
              (name, stats['triangles'], stats['max_beta'], 180.0 - stats['max_beta'],
               stats['steep_area'], stats['nonmanifold']))


# ---- рефакторинг: голый вызов для одного объекта ---------------------------

class BREKKA_OT_heightfield(Operator):
    bl_idname = "brekka.heightfield"
    bl_label = "Пересобрать (heightfield)"
    bl_description = "Пересобрать активную модель: убрать поднутрения и крутые стенки"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = _settings(context)
        obj = _mesh_target(self, context)
        if obj is None:
            return {'CANCELLED'}
        try:
            new_obj, stats = heightfield_draft(
                obj, settings.wall_angle, settings.hf_resolution,
                keep_original=settings.hf_keep_original,
                smooth_iters=settings.hf_smooth_iters)
        except RuntimeError as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        _report_analysis(self, new_obj.name, stats)
        self.report({'INFO'}, "Детали мельче ~%.2f могут сгладиться. Оригинал %s." %
                    (stats['resolution'] * 2.0,
                     "сохранён" if settings.hf_keep_original else "заменён"))
        self.report({'INFO'}, "Дальше: блок «Юбка» → «Наростить».")
        if settings.hf_keep_original:
            for o in context.scene.objects:
                o.select_set(False)
            new_obj.select_set(True)
            context.view_layer.objects.active = new_obj
        return {'FINISHED'}


# ---- анализ --------------------------------------------------------------

class BREKKA_OT_analyze(Operator):
    bl_idname = "brekka.analyze"
    bl_label = "Показать/обновить анализ"
    bl_description = ("Раскрасить активную модель: красный — круче угла/поднутрение, "
                      "оранжевый — V по углу, жёлтый — сверху перекрыто, синий — юбка")
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = _settings(context)
        obj = _mesh_target(self, context)
        if obj is None:
            return {'CANCELLED'}
        if len(obj.data.polygons) == 0:
            self.report({'ERROR'}, "В модели нет полигонов")
            return {'CANCELLED'}
        try:
            hidden = obj.hide_get() or obj.hide_viewport
        except Exception:
            hidden = obj.hide_viewport
        if hidden:
            self.report({'ERROR'}, "Объект скрыт — включите видимость")
            return {'CANCELLED'}

        kind = settings.cutter_type
        n_polys = len(obj.data.polygons)
        if n_polys > ANALYSIS_MAX_FACES:
            self.report({'ERROR'},
                        "Слишком тяжёлая модель: %d граней (лимит %d). "
                        "Упростите до ~1 млн: блок «Ремеш» → «Упростить»."
                        % (n_polys, ANALYSIS_MAX_FACES))
            return {'CANCELLED'}
        _clear_other_analysis(obj)
        started = time.time()
        try:
            red, angle, phys, total, tight = _recolor_analysis(obj, settings, context)
        except Exception as ex:
            self.report({'ERROR'}, "Анализ не удался: %s" % ex)
            return {'CANCELLED'}
        total = len(obj.data.polygons)
        heavy = total > AUTO_MAX_FACES
        names = {'BALL': "шар", 'FLAT': "плоская", 'TAPER': "конус %.2f°" % float(settings.taper_angle)}
        self.report({'INFO'},
                    "Анализ (%s Ø%.2f) за %.1f с: красных %d (круче %.1f°), "
                    "оранжевых %d (конус по углу), маджента %d (не пролезет) "
                    "из %d. Автообновление %s." %
                    (names.get(kind, kind), float(settings.analysis_diameter),
                     time.time() - started, red, settings.wall_angle,
                     angle, tight, total,
                     "вкл" if settings.analysis_auto and not heavy else "выкл"))
        if heavy:
            self.report({'INFO'},
                        "Автообновление не включено: %d граней (лимит %d). "
                        "Упростите модель до ~1 млн." % (total, AUTO_MAX_FACES))
        elif settings.analysis_auto and \
                not bpy.app.timers.is_registered(_analysis_tick):
            bpy.app.timers.register(_analysis_tick, first_interval=0.3)
        return {'FINISHED'}


class BREKKA_OT_analyze_hide(Operator):
    bl_idname = "brekka.analyze_hide"
    bl_label = "Скрыть анализ"
    bl_description = "Вернуть обычный цвет материала и остановить фоновый пересчёт"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global _analysis_visible
        _analysis_visible = False
        _stop_analysis_tick()
        _set_view_color('MATERIAL')
        return {'FINISHED'}


class BREKKA_OT_zoom_red(Operator):
    bl_idname = "brekka.zoom_red"
    bl_label = "Найти красное"
    bl_description = ("Приблизить вьюпорт к следующему красному/маджента пятну, "
                      "чтобы его поправить")
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = _mesh_target(self, context)
        if obj is None:
            return {'CANCELLED'}
        st = _analysis_state
        if not st or st.get('obj') != obj.name or \
                not len(st.get('face_colors', ())):
            self.report({'ERROR'}, "Сначала нажми «Анализ»")
            return {'CANCELLED'}
        fc = st['face_colors']
        red = np.all(fc == _PALETTE[3], axis=1)
        mag = np.all(fc == _PALETTE[4], axis=1)
        mask = red if red.any() else mag
        if not mask.any():
            self.report({'INFO'}, "Красных и маджента граней нет")
            return {'FINISHED'}
        idx = np.nonzero(mask)[0]
        me = obj.data
        mw = obj.matrix_world
        polys = me.polygons
        world = np.array([tuple(mw @ polys[int(i)].center) for i in idx],
                         dtype=np.float64)
        diag = float(obj.dimensions.length) or 1.0
        cell = max(diag / 40.0, 1e-4)
        groups = {}
        for k, w in enumerate(world.tolist()):
            key = (int(w[0] // cell), int(w[1] // cell), int(w[2] // cell))
            groups.setdefault(key, []).append(k)
        clusters = sorted(groups.values(), key=len, reverse=True)

        global _zoom_cursor
        sel = _zoom_cursor % len(clusters)
        _zoom_cursor = sel + 1
        members = world[clusters[sel]]
        center = members.mean(axis=0)
        radius = float(np.linalg.norm(members - center, axis=1).max()) \
            if len(members) > 1 else cell

        space = None
        sd = getattr(context, "space_data", None)
        if sd is not None and getattr(sd, "type", None) == 'VIEW_3D':
            space = sd
        if space is None:
            screen = getattr(context, "screen", None)
            if screen:
                for area in screen.areas:
                    if area.type == 'VIEW_3D':
                        space = area.spaces.active
                        break
        if space is not None and getattr(space, "region_3d", None) is not None:
            r3d = space.region_3d
            r3d.view_location = Vector(center)
            r3d.view_distance = max(radius * 3.0, cell * 4.0)
        self.report({'INFO'}, "Пятно %d/%d: граней %d" %
                    (sel + 1, len(clusters), len(clusters[sel])))
        return {'FINISHED'}


class _CutterButton(Operator):
    """Общий предок трёх кнопок-фрез: ставит тип и перекрашивает, если надо."""
    bl_options = {'REGISTER'}
    cutter = 'BALL'

    def execute(self, context):
        s = _settings(context)
        s.cutter_type = self.cutter
        obj = context.active_object
        if obj and obj.type == 'MESH' and _analysis_state and _analysis_visible:
            try:
                _recolor_analysis(obj, s, context)
            except Exception:
                pass
        return {'FINISHED'}


class BREKKA_OT_cutter_ball(_CutterButton):
    bl_idname = "brekka.cutter_ball"
    bl_label = "Шар"
    bl_description = "Шаровая фреза: радиус скругления как у шара"
    cutter = 'BALL'


class BREKKA_OT_cutter_flat(_CutterButton):
    bl_idname = "brekka.cutter_flat"
    bl_label = "Плоская"
    bl_description = "Плоская фреза с острой кромкой"
    cutter = 'FLAT'


class BREKKA_OT_cutter_taper(_CutterButton):
    bl_idname = "brekka.cutter_taper"
    bl_label = "Конусная"
    bl_description = ("Конусная сферическая Ø0.5 (R0.25, уклон 4.88°) — "
                      "проход считается оранжевым")
    cutter = 'TAPER'

    def execute(self, context):
        s = _settings(context)
        s.cutter_type = self.cutter
        obj = context.active_object
        if obj and obj.type == 'MESH' and _analysis_state and _analysis_visible:
            try:
                _recolor_analysis(obj, s, context)
            except Exception:
                pass
        return {'FINISHED'}


def _recolor_analysis(obj, s, context):
    """Пересчитать раскраску активной модели под текущие настройки."""
    d_ball = float(s.d_ball)
    d_flat = float(s.d_flat)
    d_taper = float(s.d_taper)
    min_radius = min(d_ball, d_flat, d_taper) * 0.5
    res = _analysis_full(obj, float(s.analysis_diameter) * 0.5, _min_nz(s),
                         context, s.cutter_type, min_radius,
                         bool(s.check_draft), bool(s.check_tool),
                         float(s.taper_angle))
    # любой путь анализа (кнопка, фреза, юбка, сглаживание) поднимает таймер
    if s.analysis_auto and _analysis_visible and \
            not bpy.app.timers.is_registered(_analysis_tick):
        bpy.app.timers.register(_analysis_tick, first_interval=0.3)
    return res


class BREKKA_OT_skirt_grow(Operator):
    bl_idname = "brekka.skirt_grow"
    bl_label = "Наростить юбку"
    bl_description = "Опустить спинку модели вниз на заданную величину (жертвенная юбка)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = _settings(context)
        obj = _mesh_target(self, context)
        if obj is None:
            return {'CANCELLED'}
        depth = float(settings.hf_base_extend)
        if depth <= 0:
            self.report({'ERROR'}, "Задайте «Удлинить дно» больше нуля")
            return {'CANCELLED'}
        try:
            result = _grow_skirt(obj, depth)
        except Exception as ex:
            self.report({'ERROR'}, "Юбка не наросла: %s" % ex)
            return {'CANCELLED'}
        if result is None:
            self.report({'WARNING'},
                        "Не нашёл плоскую спинку на самом низу — наращивать некуда")
            return {'CANCELLED'}
        n, z_back = result
        self.report({'INFO'},
                    "Юбка %.2f нарощена (спинка z=%.2f, %d вершин). "
                    "Срезать булеаном на %.2f выше нового низа." %
                    (depth, z_back, n, depth))
        return {'FINISHED'}


class BREKKA_OT_skirt_rebuild(Operator):
    bl_idname = "brekka.skirt_rebuild"
    bl_label = "Пересчитать юбку"
    bl_description = ("Выделить все нижние точки в пределах юбки и покрасить их синим")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = _settings(context)
        obj = _mesh_target(self, context)
        if obj is None:
            return {'CANCELLED'}
        me = obj.data
        if len(me.vertices) == 0:
            self.report({'ERROR'}, "В модели нет вершин")
            return {'CANCELLED'}
        depth = float(settings.hf_base_extend)
        if depth <= 0:
            self.report({'ERROR'}, "Задайте «Удлинить дно» больше нуля")
            return {'CANCELLED'}

        selected, zmin = _skirt_band_verts(obj, depth)
        band = zmin + depth

        # выделяем нижние точки (видно в Edit Mode)
        mask = np.zeros(len(me.vertices), np.int32)
        mask[selected] = 1
        me.vertices.foreach_set("select", mask)
        me.update()
        obj[SKIRT_EXT] = depth

        # пересчитываем раскраску: низ станет синим
        try:
            _recolor_analysis(obj, settings, context)
        except Exception as ex:
            self.report({'ERROR'}, "Раскраска не удалась: %s" % ex)
            return {'CANCELLED'}

        self.report({'INFO'},
                    "Юбка %.2f: выделено %d вершин ниже z=%.2f — они синие." %
                    (depth, len(selected), band))
        return {'FINISHED'}


class BREKKA_OT_remesh_target(Operator):
    bl_idname = "brekka.remesh_target"
    bl_label = "Ремешнуть до цели"
    bl_description = ("Подобрать воксель автоматически, чтобы получить примерно "
                      "заданное число вершин (по умолчанию 5 млн)")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = _settings(context)
        obj = _mesh_target(self, context)
        if obj is None:
            return {'CANCELLED'}
        target = int(settings.remesh_target) * 1_000_000
        mode = obj.mode
        if mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        size = float(obj.data.remesh_voxel_size) or 0.5
        try:
            for _ in range(2):
                obj.data.remesh_voxel_size = size
                bpy.ops.object.voxel_remesh()
                count = len(obj.data.vertices)
                if count <= 0:
                    break
                if abs(count - target) <= 0.15 * target:
                    break
                size = size * math.sqrt(count / target)
                size = min(max(size, 0.01), 1.0)
        except Exception as ex:
            self.report({'ERROR'}, "Ремеш не удался: %s" % ex)
            return {'CANCELLED'}
        finally:
            if mode != 'OBJECT':
                bpy.ops.object.mode_set(mode=mode)
        after = len(obj.data.vertices)
        self.report({'INFO'},
                    "Ремеш до %d млн: получилось %d вершин (воксель %.3f)." %
                    (int(settings.remesh_target), after,
                     float(obj.data.remesh_voxel_size)))
        return {'FINISHED'}


class BREKKA_OT_simplify(Operator):
    bl_idname = "brekka.simplify"
    bl_label = "Упростить"
    bl_description = "Упростить модель до заданного числа вершин модификатором Decimate"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = _settings(context)
        obj = _mesh_target(self, context)
        if obj is None:
            return {'CANCELLED'}
        me = obj.data
        before = len(me.vertices)
        target = int(float(settings.simplify_target) * 1_000_000)
        if before <= target:
            self.report({'INFO'}, "Уже %d вершин — упрощать не нужно." % before)
            return {'FINISHED'}
        mode = obj.mode
        if mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        mod = obj.modifiers.new("Brekka Simplify", 'DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = max(0.01, target / before)
        mod.use_collapse_triangulate = True
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception as ex:
            obj.modifiers.remove(mod)
            self.report({'ERROR'}, "Упрощение не удалось: %s" % ex)
            return {'CANCELLED'}
        finally:
            if mode != 'OBJECT':
                bpy.ops.object.mode_set(mode=mode)
        after = len(obj.data.vertices)
        self.report({'INFO'},
                    "Упрощено: %d -> %d вершин. Анализ стоит перезапустить." %
                    (before, after))
        return {'FINISHED'}


class BREKKA_OT_smooth_red(Operator):
    bl_idname = "brekka.smooth_red"
    bl_label = "Сгладить красные"
    bl_description = "Сгладить вершины красных граней (аналог прохода Smooth); при ухудшении откатывает"
    bl_options = {'REGISTER', 'UNDO'}

    def _smooth_all(self, obj, me, context):
        """Мягкий проход по всей модели, кроме дна: убирает «прыщи» после
        децимации, когда помеченных граней нет."""
        was_sculpt = obj.mode == 'SCULPT'
        saved = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", saved)
        co = saved.reshape(-1, 3)
        zmin = float(co[:, 2].min())
        idx = np.nonzero(co[:, 2] > zmin + 1e-3)[0]
        if not len(idx):
            self.report({'INFO'}, "Сглаживать нечего")
            return {'FINISHED'}
        nor0 = _mesh_face_normals(me)
        total_area = float(_mesh_face_areas(me).sum())
        if was_sculpt:
            bpy.ops.object.mode_set(mode='OBJECT')
        flips = None
        try:
            for attempt, factor in enumerate((0.5, 0.3, 0.15)):
                if attempt:
                    me.vertices.foreach_set("co", saved)
                    me.update()
                _laplacian_smooth(me, idx, factor, clamp=0.6)
                nor1 = _mesh_face_normals(me)
                area1 = _mesh_face_areas(me)
                flips = np.einsum('ij,ij->i', nor0, nor1) < 0.0
                if float(area1[flips].sum()) <= max(total_area * 1e-4, 1e-9):
                    break
            else:
                me.vertices.foreach_set("co", saved)
                me.update()
                self.report({'ERROR'},
                            "Сглаживание отменилось: грани выворачиваются")
                return {'CANCELLED'}
            bad = np.nonzero(flips)[0]
            if len(bad):
                bm = bmesh.new()
                try:
                    bm.from_mesh(me)
                    bm.faces.ensure_lookup_table()
                    nf = len(bm.faces)
                    bmesh.ops.reverse_faces(
                        bm, faces=[bm.faces[int(i)] for i in bad if i < nf])
                    bm.to_mesh(me)
                finally:
                    bm.free()
            me.update()
        finally:
            if was_sculpt and obj.mode != 'SCULPT':
                bpy.ops.object.mode_set(mode='SCULPT')
        self.report({'INFO'},
                    "Помеченных граней нет — мягко сгладил всю модель "
                    "(можно повторить или отменить Ctrl+Z)")
        return {'FINISHED'}

    def execute(self, context):
        settings = _settings(context)
        obj = _mesh_target(self, context)
        if obj is None:
            return {'CANCELLED'}
        me = obj.data
        attr = me.color_attributes.get(ANALYSIS_ATTR)
        if attr is None:
            self.report({'ERROR'}, "Сначала нажмите «Показать/обновить»")
            return {'CANCELLED'}

        n_polys = len(me.polygons)
        # атрибут может быть и в домене вершин (POINT), и старый угловой (CORNER)
        n_data = len(attr.data)
        cols = np.empty(n_data * 4, np.float32)
        attr.data.foreach_get("color", cols)
        cols = cols.reshape(-1, 4)
        is_red = (cols[:, 0] > 0.8) & (cols[:, 1] < 0.2) & (cols[:, 2] < 0.2)
        if not is_red.any():
            # помеченных нет (например, после децимации остались «прыщи») —
            # мягко проходим по всей модели, кроме дна
            return self._smooth_all(obj, me, context)

        if attr.domain == 'POINT':
            red_verts = np.nonzero(is_red)[0]
        else:
            loop_verts = np.empty(len(me.loops), np.int32)
            me.loops.foreach_get("vertex_index", loop_verts)
            red_verts = np.unique(loop_verts[is_red])

        # считаем красные ГРАНИ по кэшу анализа (та же единица, что и после)
        state = _analysis_state
        if state.get('obj') == obj.name and \
                len(state.get('face_colors', ())) == n_polys:
            fc = state['face_colors']
            red_before = int(np.sum((fc[:, 0] > 0.8) & (fc[:, 1] < 0.2) &
                                    (fc[:, 2] < 0.2)))
        else:
            red_before = len(red_verts)

        was_sculpt = obj.mode == 'SCULPT'
        saved = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", saved)
        if was_sculpt:
            bpy.ops.object.mode_set(mode='OBJECT')

        smoothed = 0
        try:
            bm = bmesh.new()
            bm.from_mesh(me)
            bm.verts.ensure_lookup_table()
            core = set()
            for i in red_verts:
                try:
                    core.add(bm.verts[int(i)])
                except Exception:
                    pass
            ring = set(core)
            frontier = set(core)
            for _ in range(2):
                nxt = set()
                for v in frontier:
                    for e in v.link_edges:
                        w = e.other_vert(v)
                        if w not in ring:
                            nxt.add(w)
                ring |= nxt
                frontier = nxt
            movable_idx = [v.index for v in (ring - frontier)]
            bm.free()
            bm = None

            # Сглаживание свободных вершин может вытянуть тонкие грани в
            # иглы и вывернуть их — в Solid они рисуются чёрными. Шаг
            # ограничиваем, а в конце пересчитываем нормали наружу.
            nor0 = _mesh_face_normals(me)
            total_area = float(_mesh_face_areas(me).sum())
            chosen = 0.0
            for attempt, factor in enumerate((0.5, 0.25, 0.12)):
                if attempt:
                    me.vertices.foreach_set("co", saved)
                    me.update()
                for _ in range(2):
                    _laplacian_smooth(me, movable_idx, factor, clamp=0.5)
                nor1 = _mesh_face_normals(me)
                area1 = _mesh_face_areas(me)
                flips = np.einsum('ij,ij->i', nor0, nor1) < 0.0
                flip_area = float(area1[flips].sum())
                allowed = max(total_area * 1e-4, 1e-9)
                if flip_area <= allowed:
                    chosen = factor
                    break
            if not chosen:
                me.vertices.foreach_set("co", saved)
                me.update()
                self.report({'ERROR'},
                            "Сглаживание отменено: грани выворачиваются")
                return {'CANCELLED'}
            # точечно разворачиваем только вывернутые грани: глобальный
            # пересчёт нормалей на открытой сетке может навредить
            bad = np.nonzero(flips)[0]
            if len(bad):
                bm2 = bmesh.new()
                try:
                    bm2.from_mesh(me)
                    bm2.faces.ensure_lookup_table()
                    nf = len(bm2.faces)
                    bmesh.ops.reverse_faces(
                        bm2, faces=[bm2.faces[int(i)] for i in bad if i < nf])
                    bm2.to_mesh(me)
                finally:
                    bm2.free()
            smoothed = len(movable_idx)
            me.update()
        finally:
            if was_sculpt and obj.mode != 'SCULPT':
                bpy.ops.object.mode_set(mode='SCULPT')

        try:
            red_after = _recolor_analysis(obj, settings, context)[0]
        except Exception:
            red_after = red_before

        reverted = red_after > red_before
        if reverted:
            me.vertices.foreach_set("co", saved)
            me.update()
            try:
                red_after = _recolor_analysis(obj, settings, context)[0]
            except Exception:
                red_after = red_before

        msg = "Сглажено %d вершин: красных было %d, стало %d." % (
            smoothed, red_before, red_after)
        if reverted:
            msg += " Стало хуже — откатил, попробуйте кистью вручную."
        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ===========================================================================
#  5. Панель
# ===========================================================================

class BREKKA_PT_panel(Panel):
    bl_label = "Brekka · фаска и анализ"
    bl_idname = "BREKKA_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Brekka"

    def draw(self, context):
        try:
            self._draw(context)
        except Exception as ex:
            self.layout.label(text="Brekka: ошибка панели", icon='ERROR')
            self.layout.label(text=str(ex)[:120])
            import traceback
            traceback.print_exc()

    @staticmethod
    def _draw_cutters(box, s):
        """Три кнопки-иконки фрезы + ползунок диаметра в той же строке."""
        row = box.row(align=True)
        row.scale_y = 1.5
        for kind, icon, opname in CUTTERS:
            row.operator(opname, text="", icon=icon,
                         depress=(s.cutter_type == kind))
        row.prop(s, "analysis_diameter", text="Ø", slider=True)

    def _draw(self, context):
        s = _settings(context)
        layout = self.layout

        # --- Пересборка ---
        box = layout.box()
        row = box.row()
        row.prop(s, "show_hf", text="",
                 icon='TRIA_DOWN' if s.show_hf else 'TRIA_RIGHT',
                 emboss=False)
        row.label(text="1. Пересборка", icon='MOD_REMESH')
        if s.show_hf:
            col = box.column(align=True)
            col.prop(s, "wall_angle")
            col.label(text="90 = вертикально, 91.5 = фаска 1.5°")
            col.prop(s, "hf_resolution")
            col.prop(s, "hf_smooth_iters")
            col.prop(s, "hf_keep_original")
            row = box.row()
            row.scale_y = 1.5
            row.operator("brekka.heightfield", text="Пересобрать",
                         icon='COLOR_BLUE', depress=True)

        # --- Юбка ---
        box = layout.box()
        row = box.row()
        row.prop(s, "show_skirt", text="",
                 icon='TRIA_DOWN' if s.show_skirt else 'TRIA_RIGHT',
                 emboss=False)
        row.label(text="2. Юбка", icon='MOD_SOLIDIFY')
        if s.show_skirt:
            box.prop(s, "hf_base_extend")
            row = box.row(align=True)
            row.operator("brekka.skirt_grow", text="Наростить", icon='MOD_SOLIDIFY')
            row.operator("brekka.skirt_rebuild", text="Пересчитать",
                         icon='FILE_REFRESH')

        # --- Ремеш ---
        box = layout.box()
        row = box.row()
        row.prop(s, "show_remesh", text="",
                 icon='TRIA_DOWN' if s.show_remesh else 'TRIA_RIGHT',
                 emboss=False)
        row.label(text="3. Ремеш", icon='MESH_ICOSPHERE')
        if s.show_remesh:
            row = box.row(align=True)
            row.prop(s, "remesh_target")
            row.operator("brekka.remesh_target", text="Ремеш", icon='MOD_REMESH')
            row = box.row(align=True)
            row.prop(s, "simplify_target")
            row.operator("brekka.simplify", text="Упростить", icon='MOD_DECIM')

        # --- Анализ ---
        box = layout.box()
        row = box.row()
        row.prop(s, "show_analysis", text="",
                 icon='TRIA_DOWN' if s.show_analysis else 'TRIA_RIGHT',
                 emboss=False)
        row.label(text="4. Анализ", icon='VIEWZOOM')
        if s.show_analysis:
            row = box.row()
            row.scale_y = 1.7
            row.operator("brekka.analyze", text="Анализ", icon='COLOR_GREEN',
                         depress=True)
            box.label(text="Проблемные зоны станут цветными.")
            self._draw_cutters(box, s)
            if s.cutter_type == 'TAPER':
                box.prop(s, "taper_angle")
            box.label(text="Что проверять:", icon='CHECKMARK')
            col = box.column(align=True)
            col.prop(s, "check_draft", text="Уклоны — круче угла стены")
            col.prop(s, "check_tool", text="Проход фрезы — где не пройдёт")
            box.prop(s, "analysis_auto")
            obj = context.active_object
            if obj and obj.type == 'MESH' and \
                    len(obj.data.polygons) > AUTO_MAX_FACES:
                box.label(text="Авто выключено: тяжёлая сетка", icon='INFO')
            row = box.row(align=True)
            row.operator("brekka.analyze_hide", text="Скрыть", icon='HIDE_OFF')
            row.operator("brekka.zoom_red", text="Найти точку",
                         icon='VIEWZOOM')
            row.operator("brekka.smooth_red", text="Сгладить",
                         icon='COLOR_RED')
            sub = box.column(align=True)
            sub.label(text="Красный — не хватает уклона / поднутрение")
            sub.label(text="Оранжевый — фреза не пройдёт по углу")
            sub.label(text="Маджента — щель уже фрезы")
            sub.label(text="Синий — юбка, её срезать")


# ===========================================================================
#  6. Регистрация
# ===========================================================================

_classes = (
    BrekkaSettings,
    BREKKA_OT_heightfield,
    BREKKA_OT_analyze,
    BREKKA_OT_analyze_hide,
    BREKKA_OT_zoom_red,
    BREKKA_OT_cutter_ball,
    BREKKA_OT_cutter_flat,
    BREKKA_OT_cutter_taper,
    BREKKA_OT_skirt_grow,
    BREKKA_OT_skirt_rebuild,
    BREKKA_OT_remesh_target,
    BREKKA_OT_simplify,
    BREKKA_OT_smooth_red,
    BREKKA_PT_panel,
)


def register():
    for cls in _classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.brekka = bpy.props.PointerProperty(type=BrekkaSettings)
    if _on_file_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_file_load)


def unregister():
    _stop_analysis_tick()
    if _on_file_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_file_load)
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    if hasattr(bpy.types.Scene, "brekka"):
        del bpy.types.Scene.brekka
