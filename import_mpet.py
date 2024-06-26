from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import array
import os
import pathlib
import time
import bpy

from math import pi, radians, atan2
from collections import namedtuple, defaultdict
from bpy_extras.image_utils import load_image
from mathutils import Matrix, Vector, Quaternion

from .pet import Puppet
from .util import (
    add_box,
    make_bbox,
    KeyFrame,
)


# Attempts to find PangYa's texture_dds folder, containing DDS textures.
def find_parent_folder(dirname, folder):
    parent = os.path.dirname(dirname)
    texdir = os.path.join(dirname, folder)
    if os.path.exists(texdir):
        return texdir
    elif dirname != parent:
        return find_parent_folder(parent, folder)
    else:
        return None


def makeMatrixFrom3x4(m):
    if m is None:
        return Matrix.Identity(4)
    return Matrix([
            [m[0], m[3], m[6], m[9]],
            [m[1], m[4], m[7], m[10]],
            [m[2], m[5], m[8], m[11]],
            [0, 0, 0, 1],
        ])


def calc_bonemat(model, bone_name, bonematrix=None):
    if bonematrix:
        return next((Matrix(b['matrix_i']) for i,b in bonematrix.items() if b['name'] == bone_name), Matrix.Identity(4))
    
    bonemat = Matrix.Identity(4)
    boneptr = next((b for b in model.bones if b.name.decode('utf-8') == bone_name), None)
    while boneptr:
        m = boneptr.matrix
        bonemat = Matrix([
            [m[0], m[3], m[6], m[9]],
            [m[1], m[4], m[7], m[10]],
            [m[2], m[5], m[8], m[11]],
            [0, 0, 0, 1],
        ]) @ bonemat
        if boneptr.parent == 255:
            break
        boneptr = model.bones[boneptr.parent]
    return bonemat

def vec_roll_to_mat3(vec, roll):
    #port of the updated C function from armature.c
    #https://developer.blender.org/T39470
    #note that C accesses columns first, so all matrix indices are swapped compared to the C version

    nor = vec.normalized()
    THETA_THRESHOLD_NEGY = 1.0e-9
    THETA_THRESHOLD_NEGY_CLOSE = 1.0e-5

    #create a 3x3 matrix
    bMatrix = Matrix().to_3x3()

    theta = 1.0 + nor[1]

    if (theta > THETA_THRESHOLD_NEGY_CLOSE) or ((nor[0] or nor[2]) and theta > THETA_THRESHOLD_NEGY):

        bMatrix[1][0] = -nor[0]
        bMatrix[0][1] = nor[0]
        bMatrix[1][1] = nor[1]
        bMatrix[2][1] = nor[2]
        bMatrix[1][2] = -nor[2]
        if theta > THETA_THRESHOLD_NEGY_CLOSE:
            #If nor is far enough from -Y, apply the general case.
            bMatrix[0][0] = 1 - nor[0] * nor[0] / theta
            bMatrix[2][2] = 1 - nor[2] * nor[2] / theta
            bMatrix[0][2] = bMatrix[2][0] = -nor[0] * nor[2] / theta

        else:
            #If nor is too close to -Y, apply the special case.
            theta = nor[0] * nor[0] + nor[2] * nor[2]
            bMatrix[0][0] = (nor[0] + nor[2]) * (nor[0] - nor[2]) / -theta
            bMatrix[2][2] = -bMatrix[0][0]
            bMatrix[0][2] = bMatrix[2][0] = 2.0 * nor[0] * nor[2] / theta

    else:
        #If nor is -Y, simple symmetry by Z axis.
        bMatrix = Matrix().to_3x3()
        bMatrix[0][0] = bMatrix[1][1] = -1.0

    #Make Roll matrix
    rMatrix = Matrix.Rotation(roll, 3, nor)

    #Combine and output result
    mat = rMatrix * bMatrix
    return mat

def mat4_to_vec_roll(mat):
    #this hasn't changed
    mat = mat.to_3x3()
    vec = mat.col[1]
    vecmat = vec_roll_to_mat3(mat.col[1], 0)
    vecmatinv = vecmat.inverted_safe()
    rollmat = vecmatinv * mat
    roll = atan2(rollmat[0][2], rollmat[2][2])
    return vec, roll

def find_texture_in_children_folder(dirname, texture):
    if pathlib.Path(os.path.join(dirname, texture)).exists():
        return dirname
    children = next(os.walk(dirname))[1]
    if len(children) == 0:
        return None
    child_dir = None
    for child in children:
        child_dir = find_texture_in_children_folder("{}/{}".format(dirname, child), texture)
        if child_dir is not None:
            break
    return child_dir

def find_texture(dirname, texture):
    if pathlib.Path(os.path.join(dirname, texture)).exists():
        return "{}/{}".format(dirname, texture)
    image_path = None
    base, ext = os.path.splitext(texture)
    if ext == ".dds":
        image_path = find_parent_folder(dirname, 'texture_dds')
        if image_path is not None:
            image_path = find_texture_in_children_folder(image_path, texture)
    else:
        for folder in ('z_common', 'map_source', 'misc', 'effect', 'texture_dds',):
            image_path = find_parent_folder(dirname, folder)
            if image_path is not None:
                image_path = find_texture_in_children_folder(image_path, texture)
                if image_path is not None:
                    break
                    
    return "{}/{}".format(image_path, texture)


# [ have mask
# ][ have mask
# [ dds transparent to alpha
# ! specular material
# + unknown
# ] unknown
def import_material_v280(mtrl, filepath):

    dirname = os.path.dirname(filepath)
    fn = mtrl.fn.decode('utf-8')

    # construct material
    new_mtrl = bpy.data.materials.new(fn)
    new_mtrl.use_nodes = True
    output_node = new_mtrl.node_tree.nodes.get("Material Output")
    if output_node is None:
        output_node = new_mtrl.node_tree.nodes.new("ShaderNodeOutputMaterial")

    # remove unused nodes
    nodes_to_remove = [n for n in new_mtrl.node_tree.nodes if n != output_node]
    for n in nodes_to_remove:
        new_mtrl.node_tree.nodes.remove(n)

    new_image = None
    if fn is not None:
        # open image
        path = pathlib.Path(fn)
        if path.is_absolute():
            image_path = str(path)
        else:
            image_path = find_texture(dirname, fn)
            if not pathlib.Path(image_path).exists():
                print("(texture) {} file not found".format(image_path))
        if pathlib.Path(image_path).exists():
            base, ext = os.path.splitext(image_path)
            filename = os.path.basename(image_path)
            if ext == '.dds' and filename[0] == '[':
                # mask texture
                new_image = bpy.data.images.load(image_path)
                
                # BSDF Principled
                bsdfpre_node = new_mtrl.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
                bsdfpre_node.location[1] = output_node.location[1]
                bsdfpre_node.inputs["Specular"].default_value = 0
                bsdfpre_node.inputs[9].default_value = 0.929
                bsdfpre_node.inputs[13].default_value = 0.5
                bsdfpre_node.inputs[16].default_value = 1

                #make texture node
                texture_node = new_mtrl.node_tree.nodes.new("ShaderNodeTexImage")
                texture_node.location[1] = output_node.location[1]
                texture_node.image = new_image
                texture_node.projection = 'FLAT'

                #make mixer node
                mixer_node = new_mtrl.node_tree.nodes.new("ShaderNodeMixShader")
                mixer_node.location[1] = output_node.location[1]
                
                #make transparent node
                trans_node = new_mtrl.node_tree.nodes.new("ShaderNodeBsdfTransparent")
                trans_node.location[1] = output_node.location[1]

                new_mtrl.node_tree.links.new(bsdfpre_node.inputs["Base Color"], texture_node.outputs["Color"])
                new_mtrl.node_tree.links.new(mixer_node.inputs[0], texture_node.outputs["Alpha"])
                new_mtrl.node_tree.links.new(mixer_node.inputs[1], trans_node.outputs["BSDF"])
                new_mtrl.node_tree.links.new(mixer_node.inputs[2], bsdfpre_node.outputs["BSDF"])
                new_mtrl.node_tree.links.new(output_node.inputs["Surface"], mixer_node.outputs["Shader"])

                new_mtrl.blend_method = 'CLIP'
                new_mtrl.show_transparent_back = False
                new_mtrl.shadow_method = 'CLIP'
                new_mtrl.alpha_threshold = 0.4
            elif (filename[0] == '[' or (filename[0] == ']' and filename[1] == '[')) and pathlib.Path("{}_mask{}".format(base, ext)).exists():
                # mask texture
                new_image = bpy.data.images.load(image_path)
                mask_image = bpy.data.images.load("{}_mask{}".format(base, ext))

                # BSDF Principled
                bsdfpre_node = new_mtrl.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
                bsdfpre_node.location[1] = output_node.location[1]
                bsdfpre_node.inputs["Specular"].default_value = 0
                bsdfpre_node.inputs[9].default_value = 0.929
                bsdfpre_node.inputs[13].default_value = 0.5
                bsdfpre_node.inputs[16].default_value = 1

                #make texture node
                texture_node = new_mtrl.node_tree.nodes.new("ShaderNodeTexImage")
                texture_node.location[1] = output_node.location[1]
                texture_node.image = new_image
                texture_node.projection = 'FLAT'

                #make mask node
                mask_node = new_mtrl.node_tree.nodes.new("ShaderNodeTexImage")
                mask_node.location[1] = output_node.location[1]
                mask_node.image = mask_image
                mask_node.projection = 'FLAT'

                #make mixer node
                mixer_node = new_mtrl.node_tree.nodes.new("ShaderNodeMixShader")
                mixer_node.location[1] = output_node.location[1]
                
                #make transparent node
                trans_node = new_mtrl.node_tree.nodes.new("ShaderNodeBsdfTransparent")
                trans_node.location[1] = output_node.location[1]

                new_mtrl.node_tree.links.new(bsdfpre_node.inputs["Base Color"], texture_node.outputs["Color"])
                new_mtrl.node_tree.links.new(mixer_node.inputs[0], mask_node.outputs["Color"])
                new_mtrl.node_tree.links.new(mixer_node.inputs[1], trans_node.outputs["BSDF"])
                new_mtrl.node_tree.links.new(mixer_node.inputs[2], bsdfpre_node.outputs["BSDF"])
                new_mtrl.node_tree.links.new(output_node.inputs["Surface"], mixer_node.outputs["Shader"])

                new_mtrl.blend_method = 'CLIP'
                new_mtrl.shadow_method = 'CLIP'
                new_mtrl.alpha_threshold = 0.4
            else:
                new_image = bpy.data.images.load(image_path)

                # BSDF Principled
                bsdfpre_node = new_mtrl.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
                bsdfpre_node.location[1] = output_node.location[1]
                bsdfpre_node.inputs["Specular"].default_value = 0
                bsdfpre_node.inputs[9].default_value = 0.929
                bsdfpre_node.inputs[13].default_value = 0.5
                bsdfpre_node.inputs[16].default_value = 1

                # make texture node
                texture_node = new_mtrl.node_tree.nodes.new("ShaderNodeTexImage")
                texture_node.location[1] = output_node.location[1]
                texture_node.image = new_image
                texture_node.projection = 'FLAT'
                
                new_mtrl.node_tree.links.new(output_node.inputs["Surface"], bsdfpre_node.outputs["BSDF"])
                new_mtrl.node_tree.links.new(bsdfpre_node.inputs["Base Color"], texture_node.outputs["Color"])

        new_mtrl['attr_flag'] = "%d" % int.from_bytes(mtrl.flag, byteorder='little', signed=True)
        new_mtrl['attr_group'] = "%d" % mtrl.group
        new_mtrl['attr_diffuse'] = "0x%.08x" % mtrl.diffuse
        new_mtrl['attr_handle'] = "%d" % mtrl.handle
    else:
        # make specular node
        specular_node = new_mtrl.node_tree.nodes.new("ShaderNodeEeveeSpecular")
        specular_node.location[1] = output_node.location[1]
        specular_node.inputs["Base Color"].default_value = (1, 1, 1, 1)
        specular_node.inputs["Specular"].default_value = [
            0.0, 0.0, 0.0, 0.0
        ]
        specular_node.inputs["Emissive Color"].default_value = [
            0.2, 0.2, 0.2, 0.2
        ]
        new_mtrl.node_tree.links.new(output_node.inputs["Surface"],
                                     specular_node.outputs["BSDF"])

    return {"material": new_mtrl, "image": new_image}

def load_pet(context, file, matrix, setting):
    dirname = os.path.dirname(file.name)
    filename = os.path.basename(file.name)
    
    model = Puppet()
    model.load(file)

    materials = []

    # Make materials for each texture.
    for texture in model.textures:
        mat = import_material_v280(texture, file.name)

        materials.append(mat['material'])

    if len(context.selected_objects) > 0:
        # change to object mode, to deselect all objects in collection
        bpy.ops.object.mode_set(mode='OBJECT')

        # deselect all objects in collection
        bpy.ops.object.select_all(action='DESELECT')

    # Collection
    collection = bpy.data.collections.new(filename)

    context.scene.collection.children.link(collection)

    # make armature
    def MakeArmature():
        # Armature
        armature = bpy.data.armatures.new(filename + " Armature")
        armature.display_type = 'STICK' # Large bones otherwise make this a bit ridiculous.
        armature_obj = bpy.data.objects.new(filename + " Armature", armature)
        collection.objects.link(armature_obj)
        armature_obj.select_set(True)

        # Create bones.
        context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bbonemap = {}
        bonematrix = {}

        # Pass 1: Create bones
        for id, bone in enumerate(model.bones):
            name = bone.name.decode('utf-8')
            bbone = armature.edit_bones.new(name)
            armature.edit_bones.active = bbone

            bbone.use_deform = True
            bbone.use_connect = True
            bbone.use_inherit_rotation = True
            bbone.use_inherit_scale = True
            bbone.use_local_location = True

            bbonemap[id] = bbone

        # Pass 2: Assign parents
        for id, bone in enumerate(model.bones):
            bbone = bbonemap[id]
            armature.edit_bones.active = bbone

            bonemat = matrix @ calc_bonemat(model, bbone.name)

            #tail, roll = mat4_to_vec_roll(bonemat)
            #bbone.head = bonemat.to_translation()
            #bbone.tail = tail*1 + bbone.head
            #bbone.roll = roll

            bbone.transform(bonemat)
            #bbone.matrix = bonemat

            # todo: calculate roll?
            #bbone.head = bonemat.to_translation()

            #bbone.tail = bonemat.to_translation()
            #if bone.parent == 255 or bbone.tail.length == 0.0:
            #    bbone.tail = 0,0,-0.005
            #bbone.roll = 0
            # !@
            #loc, rot, sca = bonemat.decompose()
            #print("bone({}) - loc {}, rot {}, sca {}".format(id, loc, rot, sca))

            bonematrix[str(id)] = {'id': id, 'name': bbone.name, 'matrix': bonemat, 'matrix_i': matrix.inverted() @ bonemat}

            if bone.parent != 255:
                parent = bbonemap[bone.parent]
                bbone.parent = parent
                #bbone.head = parent.tail

        bpy.ops.object.mode_set(mode='OBJECT')

        # set bonematrix into object of armature
        armature_obj['bonematrix'] = bonematrix
        
        return armature_obj, armature, bonematrix

    armature = None
    armature_obj = None
    bonematrix = {}

    # verifica se já tem o bone padrão carregado
    if model.is_mpet or model.is_apet:
        char_letter = filename.find('_')
        if char_letter != -1:
            for obj_key in filter(lambda key: key.find('.bpet Armature') >= 0, bpy.data.objects.keys()):
                char_letter_obj = obj_key.find('_')
                if char_letter_obj != -1 and char_letter == char_letter_obj:
                    armature_obj = bpy.data.objects.get(obj_key)
                    if armature_obj:
                        armature = armature_obj.data
                        bonematrix = armature_obj['bonematrix']
                        context.view_layer.objects.active = armature_obj
                        bpy.ops.object.mode_set(mode='OBJECT')
                        break

    if not armature_obj:
        if model.is_apet:
            raise Exception('bpet not loaded')
        
        armature_obj, armature, bonematrix = MakeArmature()
        print('criou um novo bone: ', armature_obj.name)

    mesh_obj = None

    # Geometry!
    for mesh in model.meshes:
        bmesh = bpy.data.meshes.new(filename)
        obj = bpy.data.objects.new(filename, bmesh)

        # Flatten vertices down to [x,y,z,x,y,z...] array.
        verts = []
        for vert in mesh.vertices:
            vector = Vector([0, 0, 0, 0])
            vertpos = Vector([vert.x, vert.y, vert.z, 1])

            weight = vert.bone_weights[0]
            mat = calc_bonemat(model, model.bones[weight.id].name.decode('utf-8'), bonematrix)
            vector += mat @ vertpos * (1.0 / 255 * weight.weight)

            verts.extend((
                (vector.x / vector.w),
                (vector.y / vector.w),
                (vector.z / vector.w),
            ))
        bmesh.vertices.add(len(mesh.vertices))
        bmesh.vertices.foreach_set("co", verts)

        # Polygons
        num_faces = len(mesh.polygons)
        bmesh.polygons.add(num_faces)
        bmesh.loops.add(num_faces * 3)
        faces = []
        for p in mesh.polygons:
            faces.extend((
                p.indices[0].index,
                p.indices[1].index,
                p.indices[2].index
            ))
        bmesh.polygons.foreach_set("loop_start", range(0, num_faces * 3, 3))
        bmesh.polygons.foreach_set("loop_total", (3,) * num_faces)
        bmesh.polygons.foreach_set("use_smooth", (True,) * num_faces)
        bmesh.loops.foreach_set("vertex_index", faces)

        # UV maps
        uvtex = bmesh.uv_layers.new()
        uvlayer = bmesh.uv_layers.active.data[:]
        for index, bpolygon in enumerate(bmesh.polygons):
            polygon = mesh.polygons[index]

            i = bpolygon.loop_start
            for index2 in polygon.indices:
                uvlayer[i].uv = index2.uvMapping[0].u, 1.0 - index2.uvMapping[0].v
                i += 1

        # Materials
        for material in materials:
            bmesh.materials.append(material)

        for index, material in enumerate(mesh.texmap):
            bmesh.polygons[index].material_index = material

        # Normals
        bmesh.create_normals_split()
        loops_nor = []
        for p in mesh.polygons:
            for index in p.indices:
                loops_nor.extend((
                    index.nx,
                    index.ny,
                    index.nz
                ))
        bmesh.loops.foreach_set("normal", loops_nor)

        bmesh.validate(clean_customdata=False) # *Very* important to not remove lnors here!
        bmesh.update()

        # Normal Set
        bpy.ops.object.shade_smooth()
        clnors = array.array('f', [0.0] * (len(bmesh.loops) * 3))
        bmesh.loops.foreach_get('normal', clnors)
        for poly in bmesh.polygons:
            poly.select = True
        bmesh.use_auto_smooth = True
        bmesh.auto_smooth_angle = 180
        bmesh.normals_split_custom_set(tuple(zip(*(iter(clnors),) * 3)))

        # Vertex groups
        VertexWeight = namedtuple('VertexWeight', ['vertex', 'weight'])
        groups = {}

        # Translate weights into structure of groups.
        for id, vertex in enumerate(mesh.vertices):
            for weight in vertex.bone_weights:
                group = groups.get(weight.id, [])
                group.append(VertexWeight(id, 1.0 / 255 * weight.weight))
                groups[weight.id] = group

        # Load groups into Blender
        for id, weights in groups.items():
            group_name = model.bones[id].name.decode('utf-8')
            bgroup = obj.vertex_groups.new(name=group_name)
            for weight in weights:
                bgroup.add([weight.vertex], weight.weight, 'ADD')

        obj.matrix_world = obj.matrix_world @ matrix
        collection.objects.link(obj)

        # Armature modifier
        bmodifier = obj.modifiers.new(armature.name, type='ARMATURE')
        bmodifier.show_expanded = False
        bmodifier.use_vertex_groups = True
        bmodifier.use_bone_envelopes = False
        bmodifier.object = armature_obj
        
        # set parent armature to obj
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
        
        # flip normals
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode='OBJECT')

        mesh_obj = obj

    # Collision Box
    if setting.collisionBox_enable and len(model.collisions) > 0:
        coll_obj = bpy.data.objects.new('Collisions', None)
        collection.objects.link(coll_obj)
        coll_obj.parent = armature_obj
        for coll in model.collisions:
            bone_name = coll.scripts[1].decode('utf-8') # Bone name
            index = next((i for i,x in enumerate(list(bonematrix.values())) if x['name'] == bone_name), 'XNotFoundX')
            if index != 'XNotFoundX':
                bone = list(bonematrix.values())[index]
                bb = make_bbox(collection, coll.scripts[0].decode('utf-8'), coll.area, bone['matrix'], coll_obj, not setting.collisionBox_show)
                bb.matrix_world = bb.matrix_world @ matrix @ Matrix.Rotation(pi/2,4,'Y')
                bb['shape'] = coll.shape
                bb['show'] = coll.show
            else:
                print("dont have bone({}) in list of bones".format(bone_name))

    # Animations
    def InitAnimation():
        if len(model.animations) <= 0:
            return
        
        # end of frame animation
        endf = context.scene.frame_end if setting.max_frame == '-1' else int(setting.max_frame)

        upAxisY = Matrix.Rotation(pi/2,4,'Y')
        upAxisX = Matrix.Rotation(pi/2,4,'X')
        mat_BlenderToSMD = Matrix.Rotation(radians(90),4,'X') @ Matrix.Rotation(radians(180),4,'Z')
        
        context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        keyframes = defaultdict(list)

        for anim in model.animations:
            try:
                bone = armature_obj.pose.bones[model.bones[anim.bone_id].name.decode('utf-8')]
                for pos in anim.positions:
                    frame = int("%.0f" % (pos.time * 30))
                    vpos = Vector((pos.position[0], pos.position[1], pos.position[2]))
                    mp = Matrix.Translation(vpos)
                    kfi = next((i for i,kf in enumerate(keyframes[bone]) if kf.frame == frame), None)
                    if kfi:
                        keyframes[bone][kfi].pos = True
                        keyframes[bone][kfi].matrix = keyframes[bone][kfi].matrix @ mp
                        keyframes[bone][kfi].vpos = vpos
                    else:
                        keyframes[bone].append(KeyFrame(
                            _frame=frame,
                            _pos=True,
                            _matrix=mp,
                            _vpos=vpos
                        ))
                for rot in anim.rotations:
                    frame = int("%.0f" % (rot.time * 30))
                    qrot = Quaternion((rot.rotation[3], rot.rotation[0], rot.rotation[1], rot.rotation[2])).normalized()
                    mp = qrot.copy().to_matrix().to_4x4()
                    kfi = next((i for i,kf in enumerate(keyframes[bone]) if kf.frame == frame), None)
                    if kfi:
                        keyframes[bone][kfi].rot = True
                        keyframes[bone][kfi].matrix = mp @ keyframes[bone][kfi].matrix
                        keyframes[bone][kfi].qrot = qrot
                    else:
                        keyframes[bone].append(KeyFrame(
                            _frame=frame,
                            _rot=True,
                            _matrix=mp,
                            _qrot=qrot
                        ))
            except KeyError:
                pass

        if not armature_obj.animation_data:
            armature_obj.animation_data_create()
    
        action = bpy.data.actions.new(armature_obj.name + '_act')
        action.use_fake_user = True

        armature_obj.animation_data.action = action

        if 'fps' in dir(action):
            action.fps = 30
            context.scene.render.fps = 30
            context.scene.render.fps_base = 1

        context.scene.frame_start = 0
        context.scene.frame_end = int("%.0f" % (model.animations[-1].animTime * 30))

        for bone in armature_obj.pose.bones:
            bone.rotation_mode = 'QUATERNION'
    
        # remove duplicates
        for bone,frames in list(keyframes.items()):
            if not frames:
                del keyframes[bone]

        armature_obj.select_set(True)
        context.view_layer.objects.active = armature_obj

        bpy.ops.object.mode_set(mode='POSE')

        # apply recursive bone
        def ApplyRecursive(bone):
            keys = keyframes.get(bone)
            if keys:
                # Generate curves
                curvesLoc = None
                curvesRot = None
                bone_string = "pose.bones[\"{}\"].".format(bone.name)
                group = action.groups.new(name=bone.name)
                for keyframe in keys:
                    if curvesLoc and curvesRot: break
                    if keyframe.pos and not curvesLoc:
                        curvesLoc = []
                        for i in range(3):
                            curve = action.fcurves.new(data_path=bone_string + "location",index=i)
                            curve.group = group
                            curvesLoc.append(curve)
                    if keyframe.rot and not curvesRot:
                        curvesRot = []
                        for i in range(4):
                            curve = action.fcurves.new(data_path=bone_string + "rotation_quaternion",index=i)
                            curve.group = group
                            curvesRot.append(curve)
                
                # apply
                for keyframe in keys:

                    if keyframe.pos:
                        for i in range(3):
                            curvesLoc[i].keyframe_points.add(1)
                            curvesLoc[i].keyframe_points[-1].co = [keyframe.frame,keyframe.vpos[i]]

                    if keyframe.rot:
                        for i in range(4):
                            curvesRot[i].keyframe_points.add(1)
                            curvesRot[i].keyframe_points[-1].co = [keyframe.frame,keyframe.qrot[i]]

                    context.view_layer.update()
                    
                    # limit load frames
                    if keyframe.frame > endf:
                        break
            # Recurse
            for child in bone.children:
                ApplyRecursive(child)

        # Start
        for bone in armature_obj.pose.bones:
            if not bone.parent:
                ApplyRecursive(bone)

        context.scene.frame_set(0)

        for fc in action.fcurves:
            fc.update()

        for bone in armature_obj.pose.bones:
            bone.location.zero()
            bone.rotation_quaternion.identity()
        scn = context.scene

        if scn.frame_current == 1:
            scn.frame_set(0)
        else:
            scn.frame_set(scn.frame_current)

        bpy.ops.object.mode_set(mode='OBJECT')

    if setting.anim_enable:
        InitAnimation()

    armature_obj.hide_render = True
    armature_obj.hide_set(True)

    bpy.ops.object.select_all(action='DESELECT')

def load(operator, context, filepath, matrix):
    time1 = time.process_time()
    print('importing pangya model: %r' % (filepath))

    with open(filepath, 'rb') as file:
        load_pet(context, file, matrix, operator)

    print('import done in %.4f sec.' % (time.process_time() - time1))

    return {'FINISHED'}
