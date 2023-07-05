# utils
import bpy
import bmesh
from mathutils import Vector, Matrix, Quaternion

# from blender templates
def add_box(width, height, depth):
    """
    This function takes inputs and returns vertex and face arrays.
    no actual mesh data creation is done here.
    """

    verts = [(+1.0, +1.0, -1.0),
             (+1.0, -1.0, -1.0),
             (-1.0, -1.0, -1.0),
             (-1.0, +1.0, -1.0),
             (+1.0, +1.0, +1.0),
             (+1.0, -1.0, +1.0),
             (-1.0, -1.0, +1.0),
             (-1.0, +1.0, +1.0),
             ]

    faces = [(0, 1, 2, 3),
             (4, 7, 6, 5),
             (0, 4, 5, 1),
             (1, 5, 6, 2),
             (2, 6, 7, 3),
             (4, 0, 3, 7),
            ]

    # apply size
    for i, v in enumerate(verts):
        verts[i] = v[0] * width, v[1] * depth, v[2] * height

    return verts, faces

def make_bbox(collection, name, bbox, matrix, parent=None, hide=True):

    min = Vector((bbox[0], bbox[2], bbox[1], 1.0))
    max = Vector((bbox[3], bbox[5], bbox[4], 1.0))

    loc = matrix.to_translation()
    quat = matrix.to_quaternion()

    verts_loc, faces = add_box((max.x-min.x)/2, (max.z-min.z)/2, (max.y-min.y)/2)
    mesh = bpy.data.meshes.new(name + ".mesh")
    obj = bpy.data.objects.new(name, mesh)
    bm = bmesh.new()
    for v_co in verts_loc:
        bm.verts.new(v_co)

    bm.verts.ensure_lookup_table()

    for f_idx in faces:
        bm.faces.new([bm.verts[i] for i in f_idx])

    bm.to_mesh(mesh)
    mesh.update()
    
    obj.display_type = 'BOUNDS'
    obj.hide_render = True

    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = quat

    obj.location.x = loc.x
    obj.location.y = loc.z + min.y
    obj.location.z = loc.y

    collection.objects.link(obj)

    obj.matrix_world = obj.matrix_world @ matrix

    if parent:
        obj.parent = parent

    obj.hide_set(hide)

    return obj

class KeyFrame:
    def __init__(self, _frame=None, _pos=False, _rot=False, _matrix=Matrix(), _vpos=Vector.Fill(3), _qrot=Quaternion(), _matrix_l=Matrix()):
        self.frame = _frame
        self.pos = _pos
        self.rot = _rot
        self.matrix = _matrix
        self.matrix_l = _matrix_l
        self.vpos = _vpos
        self.qrot = _qrot