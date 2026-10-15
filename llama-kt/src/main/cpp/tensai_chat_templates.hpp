// Jinja sources for the chat templates a model can be forced onto (TEN-159).
//
// common_chat_templates_init() takes a template *source*, not llama.cpp's
// built-in names ("llama3", "phi3", ...): those names only mean something to
// the legacy llama_chat_apply_template() path, and this app renders every
// prompt through the Jinja one. Only "chatml" is special-cased by chat.cpp. So
// the app passes a name and this table turns it into a source. These are
// condensed fallback templates (turns, roles, system folding), not the exact
// text the model repos ship as tokenizer.chat_template: no tools, no thinking.
// bos_token / eos_token are filled in from the vocab by common_chat_templates_init().
#pragma once

#include <string>

namespace tensai {

// Mistral v0.1 / v0.2 style: [INST] ... [/INST]. A system message is folded
// into the next user turn, the format has no slot for it.
static const char* const CHAT_TEMPLATE_MISTRAL =
    "{{ bos_token }}"
    "{%- set ns = namespace(sys='') %}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}{%- set ns.sys = ns.sys + message['content'] + '\\n\\n' %}"
    "{%- elif message['role'] == 'user' %}{{ '[INST] ' + ns.sys + message['content'] + ' [/INST]' }}{%- set ns.sys = '' %}"
    "{%- elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if ns.sys %}{{ '[INST] ' + ns.sys | trim + ' [/INST]' }}{%- endif %}";

static const char* const CHAT_TEMPLATE_LLAMA3 =
    "{{ bos_token }}"
    "{%- for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}";

// Gemma 1/2/3: roles are user / model, no system role (folded into the next user turn).
static const char* const CHAT_TEMPLATE_GEMMA =
    "{{ bos_token }}"
    "{%- set ns = namespace(sys='') %}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}{%- set ns.sys = ns.sys + message['content'] + '\\n\\n' %}"
    "{%- elif message['role'] == 'user' %}{{ '<start_of_turn>user\\n' + ns.sys + message['content'] | trim + '<end_of_turn>\\n' }}{%- set ns.sys = '' %}"
    "{%- elif message['role'] == 'assistant' %}{{ '<start_of_turn>model\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if ns.sys %}{{ '<start_of_turn>user\\n' + ns.sys | trim + '<end_of_turn>\\n' }}{%- endif %}"
    "{%- if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{%- endif %}";

static const char* const CHAT_TEMPLATE_PHI3 =
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}{{ '<|system|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{%- elif message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|assistant|>\\n' }}{%- endif %}";

// Name -> source. "chatml" stays a name (chat.cpp maps it itself), and anything
// else is passed through as a source of the caller's own.
inline std::string resolve_chat_template(const std::string& name_or_source) {
    if (name_or_source == "mistral") return CHAT_TEMPLATE_MISTRAL;
    if (name_or_source == "llama3")  return CHAT_TEMPLATE_LLAMA3;
    if (name_or_source == "gemma")   return CHAT_TEMPLATE_GEMMA;
    if (name_or_source == "phi3")    return CHAT_TEMPLATE_PHI3;
    return name_or_source;
}

}  // namespace tensai
