"""Runtime pieces for a converted Dory model.

'anemll.dory_chat' drives the five-component layout.  'anemll.runtime.dory_ane_runtime'
drives the per-(layer, head) program set the 24,576-token conversion emits: 102
programs, one Core ML file pair each, one MLState each.

    python -m anemll.runtime.dory_ane_runtime plan --model-dir DIR
    python -m anemll.runtime.dory_ane_runtime run  --model-dir DIR --prompt "..." --max-tokens 32

Nothing is re-exported here on purpose: importing the submodule from a package
whose __init__ already imported it makes 'python -m' print a RuntimeWarning.
"""
