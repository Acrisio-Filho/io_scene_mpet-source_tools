# mpet.py - implementation of Ntreev PangYa .mpet and .pet format.
# By John Chadwick <johnwchadwick@gmail.com>
#
# Special thanks to HSReina for their universal extractor and a few pointers.
# Also, to the developers of the original paktools and mpetmqo tool.

from io import BytesIO
from os.path import splitext
from enum import IntEnum

from .ioutil import (
    read_struct, write_struct,
    read_cstr, write_cstr,
    read_fixed_string, write_fixed_string,
    wraptext
)

class eFILE_TYPE(IntEnum):
    FT_TEXT= 1,
    FT_SMTL= 2,
    FT_BONE= 4,
    FT_ANIM= 8,
    FT_MESH= 16,
    FT_FANM= 32,
    FT_FRAM= 64,
    FT_MOTI= 128,
    FT_COLL= 256,
    FT_ALL= 511,
    FT_SKINLIST= 65536,
    FT_EXTR= 131072

class eSPECULAR_MATERIAL_TYPE(IntEnum):
    SMTL_BOOL= 0,
    SMTL_INT= 1,
    SMTL_FLOAT= 2,
    SMTL_VECTOR2= 3,
    SMTL_VECTOR3= 4,
    SMTL_VECTOR4= 5,
    SMTL_MATRIX= 6,
    SMTL_MATRIXARRAY= 7,
    SMTL_TEXTURE= 8,
    SMTL_NUM= 9

FILE_PET = eFILE_TYPE.FT_ALL
FILE_APET = eFILE_TYPE.FT_BONE | eFILE_TYPE.FT_ANIM | eFILE_TYPE.FT_FRAM | eFILE_TYPE.FT_MOTI
FILE_BPET = eFILE_TYPE.FT_BONE | eFILE_TYPE.FT_COLL | eFILE_TYPE.FT_EXTR
FILE_MPET = eFILE_TYPE.FT_TEXT | eFILE_TYPE.FT_SMTL | eFILE_TYPE.FT_BONE | eFILE_TYPE.FT_MESH | eFILE_TYPE.FT_FANM | eFILE_TYPE.FT_SKINLIST
FILE_ANIM_AND_BONE = eFILE_TYPE.FT_BONE | eFILE_TYPE.FT_ANIM

class Version:
    def __init__(self, major=None, minor=None):
        self.major=major
        self.minor=minor
        self.rest=0xFFFE

    def fullVersion(self):
        return (self.rest << 16) | (self.major << 8) | self.minor

    def load(self, file):
        self.minor, = read_struct(file, '<B')
        self.major, = read_struct(file, '<B')
        self.rest, = read_struct(file, '<H')
        if self.rest != 0xFFFE:
            print("Version.rest invalid(%d)" % self.rest)
            self.rest = 0xFFFE

    def save(self, file):
        write_struct(file, '<B', self.minor)
        write_struct(file, '<B', self.major)
        write_struct(file, '<H', self.rest)

    def __repr__(self):
        return "Version(%d.%d)" % (self.major, self.minor)


VERSION_1_0 = Version(1, 0)
VERSION_1_1 = Version(1, 1)
VERSION_1_2 = Version(1, 2)
VERSION_1_3 = Version(1, 3)

gFileType = FILE_PET
gVersion = VERSION_1_0

def setFileType(filepath):
    global gFileType
    gFileType = FILE_PET

    if filepath == '':
        return

    base, ext = splitext(filepath)

    if ext == '.pet':
        gFileType = FILE_PET
    elif ext == '.mpet':
        gFileType = FILE_MPET
    elif ext == '.apet':
        gFileType = FILE_APET
    elif ext == '.bpet':
        gFileType = FILE_BPET


def compareVersions(v1, v2):
    if v1.fullVersion() == v2.fullVersion():
        return 0
    
    if v1.major == v2.major:
        return -1 if v1.minor < v2.minor else 1

    return -1 if v1.major < v2.major else 1


class Puppet:
    def __init__(self, version=None, smtrls=None, animations=None, fanims=None, frames=None, motions=None, collisions=None, extras=None, bones=None, textures=None, meshes=None, meshid=None):
        self.version = version
        self.smtrls = smtrls
        self.animations = animations
        self.face_anims = fanims
        self.frames = frames
        self.motions = motions
        self.collisions = collisions
        self.extras = extras
        self.bones = bones
        self.textures = textures
        self.meshes = meshes
        self.meshid = meshid

    def load(self, file):
        self.bones = []
        self.textures = []
        self.meshes = []
        self.smtrls = []
        self.animations = []
        self.face_anims = []
        self.frames = []
        self.motions = []
        self.collisions = []
        self.extras = []
        self.is_mpet = True if splitext(file.name)[1] == '.mpet' else False
        self.is_apet = True if splitext(file.name)[1] == '.apet' else False

        # Reset Version to 1.0
        global gVersion
        gVersion = VERSION_1_0

        setFileType(file.name)

        while True:
            block = Block()
            block.load(file)

            # EOF
            if not block.is_valid():
                return

            # Version block
            if block.id == b'VERS':
                self.version = Version()
                self.version.load(block.stream)
                gVersion = self.version
                print(self.version)
            
            # Specular Material block
            elif block.id == b'SMTL':
                count, = read_struct(block.stream, '<I')

                for i in range(count):
                    smtl = SpecularMaterial()
                    smtl.load(block.stream)

                    self.smtrls.append(smtl)

            # Textures block
            elif block.id == b'TEXT':
                count, = read_struct(block.stream, '<I')

                for i in range(count):
                    texture = Texture()
                    texture.load(block.stream)

                    self.textures.append(texture)

            # Animations block
            elif block.id == b'ANIM':
                while True:
                    bone_id, = read_struct(block.stream, '<B')
                    if bone_id == 0xFE:
                        bone_id, = read_struct(block.stream, '<H')
                    if bone_id == 0xFF or bone_id == 0xFFFF:
                        break

                    animation = Animation(bone_id)
                    animation.load(block.stream)

                    self.animations.append(animation)

            # Bones block
            elif block.id == b'BONE':
                count, = read_struct(block.stream, '<B')
                if count == 0:
                    count, = read_struct(block.stream, '<H')

                for i in range(count):
                    bone = Bone()
                    bone.load(block.stream)

                    self.bones.append(bone)

            # Meshes block
            elif block.id == b'MESH':
                mesh = Mesh()
                mesh.load(block.stream, self.is_mpet)

                self.meshes.append(mesh)


            # Face Animations block
            elif block.id == b'FANM':
                count, = read_struct(block.stream, '<I')

                for i in range(count):
                    fanim = FaceAnimation()
                    fanim.load(block.stream)

                    self.face_anims.append(fanim)
            
            # Frames block
            elif block.id == b'FRAM':
                count, = read_struct(block.stream, '<I')

                for i in range(count):
                    frame = Frame()
                    frame.load(block.stream, count)

                    self.frames.append(frame)

            # Motions block
            elif block.id == b'MOTI':
                count, = read_struct(block.stream, '<I')

                for i in range(count):
                    motion = Motion()
                    motion.load(block.stream)

                    self.motions.append(motion)

            # Collisions block
            elif block.id == b'COLL':
                count, = read_struct(block.stream, '<I')

                for i in range(count):
                    collision = Collision()
                    collision.load(block.stream)

                    self.collisions.append(collision)

            # Extras block
            elif block.id == b'EXTR':
                count, = read_struct(block.stream, '<I')

                for i in range(count):
                    extra = Extra()
                    extra.load(block.stream)

                    self.extras.append(extra)

            # Unknown block
            else:
                print('warning: unknown block type: %s' % block.id)

    def save(self, file):
        self.is_mpet = True if splitext(file.name)[1] == '.mpet' else False

        setFileType(file.name)

        # Textures block
        block = Block(b'TEXT')
        write_struct(block.stream, '<I', len(self.textures))

        for texture in self.textures:
            texture.save(block.stream)

        block.write(file)

        # Bones block
        block = Block(b'BONE')
        write_struct(block.stream, '<B', len(self.bones))

        for bone in self.bones:
            bone.save(block.stream)

        block.write(file)



class Block:
    def __init__(self, id=None):
        self.stream = BytesIO()
        self.id = id

    def is_valid(self):
        return self.id is not None

    def load(self, file):
        id = file.read(4)
        if id is None or len(id) == 0:
            id = None
            return
        size, = read_struct(file, '<I')
        data = file.read(size)

        self.id = id
        self.stream = BytesIO(data)

    def save(self, file):
        size = self.stream.tell()
        self.stream.seek(0)
        data = self.stream.read()

        file.write(self.id)
        write_struct(file, '<I', size)
        file.write(data)

class Extra:
    def __init__(self, bone_id=None):
        self.bone_id = bone_id

    def load(self, file):
        self.bone_id, = read_struct(file, '<B')
        if self.bone_id == 0xFE:
            self.bone_id, = read_struct(file, '<H')

    def save(self, file):
        if self.bone_id > 0xF0:
            write_struct(file, '<BH', 0xFE, self.bone_id)
        else:
            write_struct(file, '<B', self.bone_id)

    def __repr__(self):
        return "Extra(%d)" % (self.bone_id)

class Collision:
    def __init__(self, shape=None, show=None, scripts=None, area=None):
        self.shape=shape
        self.show=show
        self.scripts=scripts # Sempre uma lista de tamanho 4, {[0] - Box name, [1] - Bone name, [2:3] - Options(bound_box) %s %s}
        self.area=area

    def load(self, file):
        self.scripts = []

        self.shape, = read_struct(file, '<I')
        self.show, = read_struct(file, '<I')
        
        for i in range(4):
            script = read_fixed_string(file)
            self.scripts.append(script)

        self.area = read_struct(file, '<6f')

    def save(self, file):
        write_struct(file, '<2I', self.shape, self.show)

        for i in range(4):
            try:
                write_fixed_string(file, self.scripts[i])
            except IndexError:
                write_fixed_string(file, b'')

        write_struct(file, '<6f', *self.area)

    def __repr__(self):
        return "Collision(%d, %d, %s %s %s %s %f %f %f %f %f %f)" % (self.shape, self.show, *self.scripts, *self.area)

class Motion:
    def __init__(self, name=None, frame_start=None, frame_end=None, next_move=None, connection_method=None, connection_time=None, top_version=None):
        self.name=name
        self.frame_start=frame_start
        self.frame_end=frame_end
        self.next_move=next_move
        self.connection_method=connection_method
        self.connection_time=connection_time
        self.top_version=top_version

    def load(self, file):
        self.name = read_fixed_string(file)
        self.frame_start, = read_struct(file, '<I')
        self.frame_end, = read_struct(file, '<I')
        self.next_move = read_fixed_string(file)
        self.connection_method = read_fixed_string(file)
        self.connection_time, = read_struct(file, '<f')
        self.top_version = read_fixed_string(file)

    def save(self, file):
        write_fixed_string(file, self.name)
        write_struct(file, '<I', self.frame_start)
        write_struct(file, '<I', self.frame_end)
        write_fixed_string(file, self.next_move)
        write_fixed_string(file, self.connection_method)
        write_struct(file, '<f', self.connection_time)
        write_fixed_string(file, self.top_version)
    
    def __repr__(self):
        return "Motion(%s, %d, %d, %s, %s, %f, %s)" % (self.name, self.frame_start, self.frame_end, self.next_move, self.connection_method, self.connection_time, self.top_version)

class Frame:
    def __init__(self, index=None, messages=None, check=None):
        self.index=index
        self.messages=messages # Sempre uma lista de tamanho 3, do tipo String
        self.check=check

    def load(self, file, total):
        self.messages = []
        self.check = 0

        self.index, = read_struct(file, '<I')
        for i in range(3):
            msg = read_fixed_string(file)
            self.messages.append(msg)

        if total < 2 and self.messages[0] == '':
            return
        
        self.check, = read_struct(file, '<I')

    def save(self, file, total):
        write_struct(file, '<I', self.index)

        if self.messages[0] != '':
            self.check = 1

        for i in range(3):
            try:
                write_fixed_string(file, self.messages[i])
            except IndexError:
                write_fixed_string(file, b'')

        if total >= 2:
            write_struct(check)

    def __repr__(self):
        return "Frame(%d, %s, %d)" % (self.index, self.message, self.check)

class FaceAnimation:
    def __init__(self, group=None, name=None, material_name=None):
        self.group=group
        self.name=name
        self.material_name=material_name

    def load(self, file):
        self.group, = read_struct(file, '<B')
        self.name = read_struct(file, '<32s')[0].split(b'\x00')[0]
        self.material_name = read_struct(file, '<32s')[0].split(b'\x00')[0]

    def save(self, file):
        write_struct(file, '<B', self.group)
        write_struct(file, '<32s', self.name)
        write_struct(file, '<32s', self.material_name)

class Position:
    def __init__(self, time=None, position=None):
        self.time=time
        self.position=position

    def load(self, file):
        self.time, = read_struct(file, '<f')
        self.position = read_struct(file, '<3f')

    def save(self, file):
        write_struct(file, '<4f', self.time, *self.position)

    def __repr__(self):
        return "Position(%f, %f, %f, %f)" % (self.time, *self.position)

class Rotation:
    def __init__(self, time=None, rotation=None):
        self.time=time
        self.rotation=rotation

    def load(self, file):
        self.time, = read_struct(file, '<f')
        self.rotation = read_struct(file, '<4f')

    def save(self, file):
        write_struct(file, '<5f', self.time, *self.rotation)

    def __repr__(self):
        return "Rotation(%f, %f, %f, %f, %f)" % (self.time, *self.rotation)

class Scaling:
    def __init__(self, time=None, scaling=None):
        self.time=time
        self.scaling=scaling

    def load(self, file):
        self.time, = read_struct(file, '<f')
        self.scaling = read_struct(file, '<3f')

    def save(self, file):
        write_struct(file, '<4f', self.time, *self.scaling)

    def __repr__(self):
        return "Scaling(%f, %f, %f, %f)" % (self.time, *self.scaling)

class FlagAnim:
    def __init__(self, time=None, scale=None):
        self.time=time
        self.scale=scale

    def load(self, file):
        self.time, = read_struct(file, '<f')
        self.scale, = read_struct(file, '<f')

    def save(self, file):
        write_struct(file, '<2f', self.time, self.scale)

    def __repr__(self):
        return "FlagAnim(%f, %f)" % (self.time, self.scale)

class Animation:
    def __init__(self, bone_id=None, positions=None, rotations=None, scalings=None, flags=None, animTime=None):
        self.bone_id=bone_id
        self.positions=positions
        self.rotations=rotations
        self.scalings=scalings
        self.flags=flags
        self.animTime=animTime

    def load(self, file):
        self.positions = []
        self.rotations = []
        self.scalings = []
        self.flags = []
        self.animTime = 0.0

        count, = read_struct(file, '<I')

        for i in range(count):
            position = Position()
            position.load(file)

            self.positions.append(position)
        
        if len(self.positions) > 0:
            self.animTime = max(self.animTime, max(self.positions, key=lambda pos: pos.time).time)
        
        count, = read_struct(file, '<I')

        for i in range(count):
            rotation = Rotation()
            rotation.load(file)

            self.rotations.append(rotation)

        if len(self.rotations) > 0:
            self.animTime = max(self.animTime, max(self.rotations, key=lambda rot: rot.time).time)

        count, = read_struct(file, '<I')

        for i in range(count):
            scaling = Scaling()
            scaling.load(file)

            self.scalings.append(scaling)

        if len(self.scalings) > 0:
            self.animTime = max(self.animTime, max(self.scalings, key=lambda scal: scal.time).time)

        if compareVersions(gVersion, VERSION_1_3) >= 0:
            count, = read_struct(file, '<I')

            for i in range(count):
                flag = FlagAnim()
                flag.load(file)

                self.flags.append(flag)

            if len(self.flags) > 0:
                self.animTime = max(self.animTime, max(self.flags, key=lambda flag: flag.time).time)
    
    def save(self, file):
        if self.bone_id > 0xF0:
            write_struct(file, '<B', 0xFE)
            write_struct(file, '<H', self.bone_id)
        else:
            write_struct(file, '<B', self.bone_id)

        write_struct(file, '<I', len(self.positions))

        for position in self.positions:
            position.save(file)

        write_struct(file, '<I', len(self.rotations))

        for rotation in self.rotations:
            rotation.save(file)

        write_struct(file, '<I', len(self.scalings))

        for scaling in self.scalings:
            scaling.save(file)

        if compareVersions(gVersion, VERSION_1_3) >= 0:
            write_struct(file, '<I', len(self.flags))

            for flag in self.flags:
                flag.save(file)

    def __repr__(self):
        return "Animations()"


class SpecularValue:
    def __init__(self, name=None, type=None, value=None):
        self.name=name
        self.type=type
        self.value=value

    def load(self, file):
        self.name = read_fixed_string(file)
        self.type, = read_struct(file, '<I')

        if self.type == eSPECULAR_MATERIAL_TYPE.SMTL_BOOL:
            self.value, = read_struct(file, '<I')
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_INT:
            self.value, = read_struct(file, '<i')
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_FLOAT:
            self.value, = read_struct(file, '<f')
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_VECTOR2:
            self.value = read_struct(file, '<2f')
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_VECTOR3:
            self.value = read_struct(file, '<3f')
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_VECTOR4:
            self.value = read_struct(file, "<4f")
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_MATRIX:
            self.value = read_struct(file, "<16f")
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_TEXTURE:
            self.value = read_fixed_string(file)
    
    def save(self, file):
        write_fixed_string(file, self.name)
        write_struct(file, '<I', self.type)

        if self.type == eSPECULAR_MATERIAL_TYPE.SMTL_BOOL:
            write_struct(file, '<I', self.value)
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_INT:
            write_struct(file, '<i', self.value)
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_FLOAT:
            write_struct(file, '<f', self.value)
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_VECTOR2:
            write_struct(file, '<2f', *self.value)
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_VECTOR3:
            write_struct(file, '<3f', *self.value)
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_VECTOR4:
            write_struct(file, '<4f', *self.value)
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_MATRIX:
            write_struct(file, '<16f', *self.value)
        elif self.type == eSPECULAR_MATERIAL_TYPE.SMTL_TEXTURE:
            write_fixed_string(file, self.value)

    def __repr__(self):
        return ("SpecularValue(%s, %d" % (self.name, self.type)) + self.value


class SpecularMaterial:
    def __init__(self, name=None, smtlvs=None):
        self.name=name
        self.smtlvs=smtlvs

    def load(self, file):
        self.smtlvs = []

        self.name = read_fixed_string(file)
        count, = read_struct(file, '<I')

        for i in range(count):
            sv = SpecularValue()
            sv.load(file)

            self.smtlvs.append(sv)

    def save(self, file):
        write_fixed_string(file, self.name)
        write_struct(file, '<I', len(self.smtlvs))
        for sv in self.smtlvs:
            sv.save(file)

    def __repr__(self):
        return "SpacularMaterial(%d)" % (len(self.smtlvs))

class Texture:
    def __init__(self, fn=None, flag=None, group=None, diffuse=None, handle=None):
        self.fn = fn
        self.flag = flag
        self.group = group
        self.diffuse = diffuse
        self.handle = handle

    def is_valid(self):
        return self.fn is not None

    def load(self, file):
        # looks like Ntreev left us some stack garbage
        self.fn = read_struct(file, '<32s')[0].split(b'\x00')[0]
        self.flag, = read_struct(file, '<c')
        self.group, = read_struct(file, '<B')
        file.read(2) # align 4
        self.diffuse, = read_struct(file, '<I')
        self.handle, = read_struct(file, '<I')

    def save(self, file):
        write_struct(file, '<32s', self.fn)
        write_struct(file, '<c', self.flag)
        write_struct(file, '<B', self.group)
        file.write("\x00\x00") # align 4
        write_struct(file, '<I', self.diffuse)
        write_struct(file, '<I', self.handle)

    def __repr__(self):
        return "Texture(%s)" % (self.fn)


class Bone:
    def __init__(self, name=None, parent=None, matrix=None, float_v13=None):
        self.name = name
        self.parent = parent
        self.matrix = matrix
        self.float_v13 = float_v13

    def is_valid(self):
        return self.name is not None

    def load(self, file):
        self.name = read_cstr(file)
        self.parent, = read_struct(file, '<B')
        if self.parent == 0xFE:
            self.parent, = read_struct(file, '<H')
        if gFileType != FILE_APET and (gFileType == FILE_PET or gFileType == FILE_BPET or gFileType == FILE_MPET):
            self.matrix = read_struct(file, '<12f')
            if compareVersions(gVersion, VERSION_1_3) >= 0:
                self.float_v13, = read_struct(file, "<f")

    def save(self, file):
        write_cstr(file, self.name)
        if self.parent > 0xF0:
            write_struct(file, '<BH', 0xFE, self.parent)
        else:
            write_struct(file, '<B', self.parent)
        if gFileType != FILE_APET and (gFileType == FILE_PET or gFileType == FILE_BPET or gFileType == FILE_MPET):
            write_struct(file, '<12f', *self.matrix)
            if compareVersions(gVersion, VERSION_1_3) >= 0:
                write_struct(file, '<f', self.float_v13)

    def __repr__(self):
        fmt = "Bone(%s, %d, [%f" + (", %f" * 11) + "])"
        return fmt % (self.name, self.parent, *self.matrix)


class BoneWeight:
    def __init__(self, weight=None, id=None):
        self.weight = weight
        self.id = id

    def load(self, file):
        self.weight, self.id = read_struct(file, '<2B')
        if self.id == 0xFE:
            self.id, = read_struct(file, '<H')

    def save(self, file):
        write_struct(file, '<B', self.weight)
        if self.id > 0xF0:
            write_struct(file, '<B', 0xFE)
            write_struct(file, '<H', self.id)
        else:
            write_struct(file, '<B', self.id)

    def __repr__(self):
        return "BoneWeight(weight=%d, id=%d)" % (self.weight, self.id)

class MpetExtraValue:
    def __init__(self, unknown=None, vertice_idx=None, vertice_len=None, poly_idx=None, poly_len=None):
        self.unknown=unknown
        self.vertice_idx=vertice_idx
        self.vertice_len=vertice_len
        self.poly_idx=poly_idx
        self.poly_len=poly_len
    
    def load(self, file):
        self.unknown, = read_struct(file, '<I')
        self.vertice_idx, = read_struct(file, '<I')
        self.vertice_len, = read_struct(file, '<I')
        self.poly_idx, = read_struct(file, '<I')
        self.poly_len, = read_struct(file, '<I')

    def save(self, file):
            write_struct(file, '<I', self.unknown)
            write_struct(file, '<I', self.vertice_idx)
            write_struct(file, '<I', self.vertice_len)
            write_struct(file, '<I', self.poly_idx)
            write_struct(file, '<I', self.poly_len)

    def __repr__(self):
        return "MpetExtraVelue(unknown=%d, vertice_index=%d, vertice_len=%d, polygon_index=%d, polygon_length=%d)" % (
            self.unknown, self.vertice_idx, self.vertice_len, self.poly_idx, self.poly_len
        )

class Vertex:
    def __init__(self, x=None, y=None, z=None, w=None, weight_global=None, bone_weights=None):
        self.x, self.y, self.z, self.w = x, y, z, w
        self.bone_weights = bone_weights
        self.weight_global=weight_global

    def load(self, file, is_mpet):
        wsum = 0
        self.bone_weights = []

        self.x, self.y, self.z = read_struct(file, '<3f')

        if is_mpet:
            self.weight_global, = read_struct(file, '<f')
        else:
            self.weight_global = 1.0

        # Bone weights continue until saturated.
        while wsum < 0xFF:
            weight = BoneWeight()
            weight.load(file)

            wsum += weight.weight
            self.bone_weights.append(weight)

        # Strangely, file leaves space for at least 2.
        if len(self.bone_weights) < 2:
            file.read(2)

    def save(self, file, is_mpet):
        write_struct(file, '<3f', self.x, self.y, self.z)

        if is_mpet:
            write_struct(file, '<f', self.weight_global)

        for weight in self.bone_weights:
            weight.save(file)

        if len(self.bone_weights) < 2:
            file.write("\x00\x00")

    def __repr__(self):
        return "Vertex(%f, %f, %f)" % (self.x, self.y, self.z)

class UVMapping:
    def __init__(self, u=None, v=None):
        self.u=u
        self.v=v

    def load(self, file):
        self.u, self.v = read_struct(file, '<2f')

    def save(self, file):
        write_struct(file, '<2f', self.u, self.v)

    def __repr__(self):
        return "UVMapping(%f, %f)" % (self.u, self.v)

class PolygonIndex:
    def __init__(self, index=None, nx=None, ny=None, nz=None, uvMapping=None):
        self.index = index
        self.nx, self.ny, self.nz = nx, ny, nz
        self.uvMapping = uvMapping

    def load(self, file):
        self.uvMapping = []

        self.index, = read_struct(file, '<I')
        self.nx, self.ny, self.nz = read_struct(file, '<3f')
        count = 1
        if compareVersions(gVersion, VERSION_1_2) >= 0:
            count, = read_struct(file, '<B')
        for i in range(count):
            uv = UVMapping()
            uv.load(file)

            self.uvMapping.append(uv)

    def save(self, file):
        write_struct(file, '<I', self.index)
        write_struct(file, '<3f', self.nx, self.ny, self.nz)
        if compareVersions(gVersion, VERSION_1_2) >= 0:
            write_struct(file, '<B', len(self.uvMapping))

            for uv in self.uvMapping:
                uv.save(file)

        elif len(self.uvMapping):
            self.uvMapping[0].save(file)
        else:
            write_struct(file, '<2f', 0.0, 0.0)

    def __repr__(self):
        return "PolygonIndex(index=%d)" % self.index


class Polygon:
    def __init__(self, indices=None):
        self.indices = indices

    def load(self, file):
        self.indices = []
        for i in range(3):
            index = PolygonIndex()
            index.load(file)

            self.indices.append(index)

    def save(self, file):
        for index in self.indices:
            index.save(file)


class Mesh:
    def __init__(self, vertices=None, polygons=None, texmap=None, mpetexs=None, map_v12=None):
        self.vertices = vertices
        self.polygons = polygons
        self.texmap = texmap
        self.mpetexs=mpetexs
        self.map_v12 = map_v12

    def load(self, file, is_mpet):
        self.vertices = []
        self.polygons = []
        self.texmap = []
        self.mpetexs = []
        self.map_v12 = []

        if is_mpet:
            num_mpetex, = read_struct(file, '<B')
            for i in range(num_mpetex):
                mpetEx = MpetExtraValue()
                mpetEx.load(file)
                print(mpetEx)
                self.mpetexs.append(mpetEx)

        # Vertices
        num_vertices, = read_struct(file, '<I')
        #print(num_vertices)
        for i in range(num_vertices):
            vertex = Vertex()
            vertex.load(file, is_mpet)
            #print(vertex.x, vertex.y, vertex.z)

            self.vertices.append(vertex)

        # Polygons
        num_polygons, = read_struct(file, '<I')
        for i in range(num_polygons):
            polygon = Polygon()
            polygon.load(file)

            self.polygons.append(polygon)

        # Mapping of polygons to textures
        for i in range(num_polygons):
            self.texmap.append(read_struct(file, '<B')[0])

        if compareVersions(gVersion, VERSION_1_2) >= 0:
            for i in range(num_polygons):
                self.map_v12.append(read_struct(file, '<B')[0])
    
    def save(self, file, is_mpet):
        if is_mpet:
            for mpetex in self.mpetexs:
                mpetex.save(file)
        write_struct(file, '<I', len(self.vertices))
        for vertex in self.vertices:
            vertex.save(file)
        write_struct(file, '<I', len(self.polygons))
        for polygon in self.polygons:
            polygon.save(file)
        for tex in self.texmap:
            write_struct(file, '<B', tex)
        if compareVersions(gVersion, VERSION_1_2) >= 0:
            for mapv12 in self.map_v12:
                write_struct(file, '<B', mapv12)

    def __repr__(self):
        return "Mesh(vertices=%s, polygons=%s)" % (
            repr(self.vertices),
            repr(self.polygons),
        )
