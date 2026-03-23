/*
 * Copyright 2000-2016 JetBrains s.r.o.
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @author: Dmitry Batkovich
 */
const navigate = (anId) => {
  const problemDiv = document.getElementById(`d${anId}`);
  const previewDiv = document.getElementById("preview");

  if (previewDiv == null) {
    return;
  }

  previewDiv.innerHTML = problemDiv != null ? problemDiv.innerHTML : "Select a problem element in tree";
};

window.navigate = navigate;

document.addEventListener("DOMContentLoaded", () => {
  // 导出报告里原本用的是内联 onclick，这里在页面加载后统一改成事件绑定，
  // 既保留原行为，也能避免编辑器把 `navigate` 误判为未使用。
  document.querySelectorAll('input[onclick^="navigate("]').forEach((element) => {
    const onclickValue = element.getAttribute("onclick");
    const match = onclickValue != null ? onclickValue.match(/navigate\((\d+)\)/) : null;

    if (match == null) {
      return;
    }

    element.addEventListener("click", () => {
      navigate(match[1]);
    });
    element.removeAttribute("onclick");
  });
});
