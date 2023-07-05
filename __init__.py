import bpy, os, pathlib
from mathutils import Matrix
from bpy_extras.io_utils import (
    ImportHelper,
    ExportHelper,
    orientation_helper,
    axis_conversion,
)

# peice of code from valve benlder tools, to reload scripts(addon)
# Python doesn't reload package sub-modules at the same time as __init__.py!
import importlib, sys
for filename in [ f for f in os.listdir(os.path.dirname(os.path.realpath(__file__))) if f.endswith(".py") ]:
	if filename == os.path.basename(__file__): continue
	module = sys.modules.get("{}.{}".format(__name__,filename[:-3]))
	if module: importlib.reload(module)

bl_info = {
    "name": "PangYa Model",
    "author": "John Chadwick",
    "version": (1, 1, 0),
    "blender": (2, 80, 0),
    "location": "File > Import-Export",
    "description": "Import-Export PangYa .pet, .apet, .bpet and .mpet files.",
    "category": "Import-Export",
    "warning": "Base from John Chadwick, with addition of new features and fixing of structures types by Acrisio Filho"
}

# ImportMpet implements the operator for importing Mpet models.
@orientation_helper(axis_forward='Z', axis_up='Y')
class ImportMpet(bpy.types.Operator, ImportHelper):
    """Import from Pangya Model (.pet, .mpet, .apet, .bpet)"""
    bl_idname = 'import_scene.pangya_mpet'
    bl_label = 'Import Pangya Model'
    bl_options = {'UNDO'}

    filename_ext = ".pet;.mpet;.apet;.bpet"
    filter_glob: bpy.props.StringProperty(
        default="*.pet;*.mpet;*.apet;*.bpet",
        options={'HIDDEN'},
    )

    anim_enable: bpy.props.BoolProperty(
        name="Import Animation",
        description="Import animation if puppet have it",
        default=True
    )

    collisionBox_enable: bpy.props.BoolProperty(
        name="Import Collision Box",
        description="Import collision box if puppet have it",
        default=False
    )

    collisionBox_show: bpy.props.BoolProperty(
        name="Show Collision Box",
        description="Show collision box in render",
        default=False
    )

    orient_to_forward: bpy.props.EnumProperty(
        name="Axis to_forward convert",
        description="Convert axis to_forward from_forward",
        items=(
            ('X',"X",""),
            ('Y',"Y",""),
            ('Z',"Z",""),
            ('-X',"-X",""),
            ('-Y',"-Y",""),
            ('-Z',"-Z",""),
        ),
        default='-X'
    )

    orient_to_up: bpy.props.EnumProperty(
        name="Axis to_up convert",
        description="Convert axis to_up from_up",
        items=(
            ('X',"X",""),
            ('Y',"Y",""),
            ('Z',"Z",""),
            ('-X',"-X",""),
            ('-Y',"-Y",""),
            ('-Z',"-Z",""),
        ),
        default='Z'
    )

    max_frame: bpy.props.EnumProperty(
        name="Max frame Animation",
        description="Número máximo de frame da animação",
        items=(
            ('-1',"NoLimit","Sem limites"),
            ('1',"1",""),
            ('2',"2",""),
            ('100',"100",""),
            ('500',"500",""),
            ('1000',"1000",""),
        ),
        default='100'
    )

    # support to load multiples files
    files: bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)

    def execute(self, context):
        from . import import_mpet

        matrix = axis_conversion(
            from_forward=self.axis_forward,
            from_up=self.axis_up,
            to_forward=self.orient_to_forward,
            to_up=self.orient_to_up,
        ).to_4x4()

        # invert axis x
        matrix = matrix @ Matrix.Scale(-1, 4, (1, 0, 0)).to_4x4()

        folder = pathlib.Path(self.filepath)
        for selection in sorted(self.files.values(), key=lambda e : -1 if os.path.splitext(e.name)[1] == '.bpet' else 0):
            fp = pathlib.Path(folder.parent, selection.name)
            if fp.suffix in self.filename_ext.split(';'):
                import_mpet.load(self, context, str(fp), matrix)
        
        return {'FINISHED'}

# ExportMpet implements the operator for exporting to Mpet.
class ExportMpet(bpy.types.Operator, ExportHelper):
    """Export to Pangya Model (.pet, .mpet, .apet, .bpet)"""
    bl_idname = 'export_scene.pangya_mpet'
    bl_label = 'Export Pangya Model'

    filename_ext = ".pet;.mpet;.apet;.bpet"
    filter_glob: bpy.props.StringProperty(
        default="*.pet;*.mpet;*.apet;*.bpet",
        options={'HIDDEN'},
    )

    use_selection = bpy.props.BoolProperty(
        name='Selection Only',
        description='Export selected objects only',
        default=False,
    )

    def execute(self, context):
        from . import export_mpet
        return export_mpet.save(self, context)


# Registration/Menu items
def menu_func_export(self, context):
    self.layout.operator(ExportMpet.bl_idname, text="PangYa Model (.pet, .mpet, .apet, .bpet)")


def menu_func_import(self, context):
    self.layout.operator(ImportMpet.bl_idname, text="PangYa Model (.pet, .mpet, .apet, .bpet)")


def register():
    bpy.utils.register_class(ImportMpet)
    bpy.utils.register_class(ExportMpet)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(ImportMpet)
    bpy.utils.unregister_class(ExportMpet)

    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()
