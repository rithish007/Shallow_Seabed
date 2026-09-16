"""Custom-depth stencil tagging (original plan Step 2) - assigns a per-class stencil ID to foliage HISM components and matching mesh actors, for use with a class-color post-process mask material (e.g. M_StencilMask: SceneTexture(PPI_CUSTOM_STENCIL) -> MaterialExpressionCustom -> Emissive, MD_POST_PROCESS, BL_REPLACING_TONEMAPPER)."""
import unreal

CLASS_STENCIL_MAP = {
    1: ("S_Thai_Beach_Corals_Pack_", "SM_Coralline"),
    2: ("SM_Seaweed", "SM_Algae", "SM_RedAlgae"),
    3: ("SM_URock",),
    4: ("SM_Sponge",),
}


def _matches(name, prefixes):
    return any(name.startswith(p) for p in prefixes)


def tag_all():
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = eas.get_all_level_actors()

    foliage_tagged = 0
    actors_tagged = 0

    for a in actors:
        cls_name = a.get_class().get_name()

        if "InstancedFoliageActor" in cls_name:
            for comp in a.get_components_by_class(unreal.InstancedStaticMeshComponent):
                mesh = comp.get_editor_property("static_mesh")
                if mesh is None:
                    continue
                mesh_name = mesh.get_name()
                for stencil_id, prefixes in CLASS_STENCIL_MAP.items():
                    if _matches(mesh_name, prefixes):
                        comp.set_editor_property("render_custom_depth", True)
                        comp.set_editor_property("custom_depth_stencil_value", stencil_id)
                        foliage_tagged += 1
                        break
            continue

        label = a.get_actor_label()
        for stencil_id, prefixes in CLASS_STENCIL_MAP.items():
            if _matches(label, prefixes):
                for comp in a.get_components_by_class(unreal.StaticMeshComponent):
                    comp.set_editor_property("render_custom_depth", True)
                    comp.set_editor_property("custom_depth_stencil_value", stencil_id)
                actors_tagged += 1
                break

    return {"foliage_tagged": foliage_tagged, "actors_tagged": actors_tagged}
