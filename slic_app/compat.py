from __future__ import annotations

def patch_streamlit_drawable_canvas_compat() -> None:
    """
    SHIM de compatibilidade: streamlit-drawable-canvas em Streamlit novo.
    (o canvas chama streamlit.elements.image.image_to_url, que mudou)
    """
    try:
        import streamlit.elements.image as st_image
        from streamlit.elements.lib.image_utils import image_to_url as _image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig

        def _image_to_url_compat(image, *args, **kwargs):
            # New signature: (image, layout_config, clamp, channels, output_format, image_id)
            if args:
                if isinstance(args[0], LayoutConfig):
                    return _image_to_url(image, *args, **kwargs)
                # Old signature: (image, width, clamp, channels, output_format, image_id)
                if len(args) >= 5:
                    width, clamp, channels, output_format, image_id = args[:5]
                    layout_config = LayoutConfig(width=width)
                    return _image_to_url(image, layout_config, clamp, channels, output_format, image_id)

            if "layout_config" in kwargs:
                return _image_to_url(image, **kwargs)

            # Fallback for keyword-style old signature
            if "width" in kwargs:
                width = kwargs.pop("width")
                clamp = kwargs.pop("clamp")
                channels = kwargs.pop("channels")
                output_format = kwargs.pop("output_format")
                image_id = kwargs.pop("image_id")
                layout_config = LayoutConfig(width=width)
                return _image_to_url(image, layout_config, clamp, channels, output_format, image_id)

            return _image_to_url(image, *args, **kwargs)

        st_image.image_to_url = _image_to_url_compat  # monkeypatch
    except Exception:
        pass
