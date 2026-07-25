# Downloads every example model next to its JSON config at configure time, so
# the examples work out of the box: each JSON's model_path is just the file
# name, which anira_tilde resolves relative to the JSON itself — users never
# edit paths. Files are fetched once and reused on later configures (delete a
# file to re-download it). The model binaries are gitignored; this download is
# their only source. Disable with -DANIRA_TILDE_DOWNLOAD_EXAMPLE_MODELS=OFF
# (e.g. offline builds — everything except the examples works without them).
#
# Include AFTER setup-dependencies: the LibTorch-only models are skipped
# unless the build opted into ANIRA_WITH_LIBTORCH.

option(ANIRA_TILDE_DOWNLOAD_EXAMPLE_MODELS
       "Download the example models next to their JSON configs at configure time" ON)

if(NOT ANIRA_TILDE_DOWNLOAD_EXAMPLE_MODELS)
    return()
endif()

# <example-dir>|<file-name>|<url>
set(_anira_tilde_example_models
    "anira_cnn_amp_emulation_onnx|steerable-nafx-libtorch-dynamic.onnx|https://github.com/faressc/steerable-nafx/raw/refs/heads/main/models/model_0/steerable-nafx-libtorch-dynamic.onnx"
    "anira_cnn_amp_emulation_executorch|steerable-nafx-executorch.pte|https://github.com/faressc/steerable-nafx/raw/refs/heads/main/models/model_0/steerable-nafx-executorch.pte"
    "anira_rnn_amp_emulation_tflite|stateful-lstm-dynamic.tflite|https://github.com/vackva/stateful-lstm/raw/refs/heads/main/models/model_0/stateful-lstm-dynamic.tflite"
    "anira_rnn_amp_emulation_executorch|stateful-lstm-executorch.pte|https://github.com/vackva/stateful-lstm/raw/refs/heads/main/models/model_0/stateful-lstm-executorch.pte"
    "rave_djembe|rave_forward.onnx|https://github.com/anira-project/example-models/raw/refs/heads/main/third-party/ircam-acids/RAVE/RaveDjembe/models/rave_forward.onnx"
    "rave_djembe|rave_forward.pte|https://github.com/anira-project/example-models/raw/refs/heads/main/third-party/ircam-acids/RAVE/RaveDjembe/models/rave_forward.pte"
    "rave_djembe|rave_encoder.onnx|https://github.com/anira-project/example-models/raw/refs/heads/main/third-party/ircam-acids/RAVE/RaveDjembe/models/rave_encoder.onnx"
    "rave_djembe|rave_encoder.pte|https://github.com/anira-project/example-models/raw/refs/heads/main/third-party/ircam-acids/RAVE/RaveDjembe/models/rave_encoder.pte"
    "rave_djembe|rave_decoder.onnx|https://github.com/anira-project/example-models/raw/refs/heads/main/third-party/ircam-acids/RAVE/RaveDjembe/models/rave_decoder.onnx"
    "rave_djembe|rave_decoder.pte|https://github.com/anira-project/example-models/raw/refs/heads/main/third-party/ircam-acids/RAVE/RaveDjembe/models/rave_decoder.pte"
)

# TorchScript models load only in LibTorch-enabled builds; don't pull the
# (large) downloads into the default configuration.
if(ANIRA_WITH_LIBTORCH)
    list(APPEND _anira_tilde_example_models
        "rave_darbouka_libtorch|darbouka_onnx.ts|https://play.forum.ircam.fr/rave-vst-api/get_model/darbouka_onnx"
    )
endif()

set(_failed "")
foreach(_entry IN LISTS _anira_tilde_example_models)
    string(REPLACE "|" ";" _parts "${_entry}")
    list(GET _parts 0 _dir)
    list(GET _parts 1 _file)
    list(GET _parts 2 _url)
    set(_dest "${CMAKE_SOURCE_DIR}/examples/${_dir}/${_file}")
    if(EXISTS "${_dest}")
        continue()
    endif()
    message(STATUS "anira_tilde: downloading example model ${_dir}/${_file}")
    file(DOWNLOAD "${_url}" "${_dest}" STATUS _st TIMEOUT 300)
    list(GET _st 0 _code)
    if(NOT _code EQUAL 0)
        list(GET _st 1 _msg)
        file(REMOVE "${_dest}")
        list(APPEND _failed "${_dir}/${_file} (${_msg})")
    endif()
endforeach()

if(_failed)
    list(JOIN _failed "\n    " _failed_str)
    message(WARNING "anira_tilde: some example models could not be downloaded — the affected "
                    "examples will not load until they are (re-run the configure to retry):\n"
                    "    ${_failed_str}")
endif()
