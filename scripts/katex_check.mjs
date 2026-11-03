// 표준 입력: 수식 문자열의 JSON 배열. 표준 출력: 같은 순서의 {ok, glyphs, error} 배열.
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { readFileSync } from "node:fs";

// 저장소에 package.json을 두지 않으려고 .cache/node에 설치한 KaTeX를 경로로 불러온다.
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(resolve(root, ".cache/node/package.json"));
const katex = require("katex");

const ENTITIES = { "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#x27;": "'" };

function glyphs(html) {
  // 늘어나는 괄호와 근호는 PDF에서도 글리프가 아닌 경로로 그려지므로 svg를 비교에서 뺀다.
  return html
    .replace(/<svg[\s\S]*?<\/svg>/g, "")
    .replace(/<[^>]+>/g, "")
    .replace(/&(?:amp|lt|gt|quot|#x27);/g, (m) => ENTITIES[m])
    .replace(/[\s​]/g, "");
}

const formulas = JSON.parse(readFileSync(0, "utf8"));
const results = formulas.map((tex) => {
  try {
    const html = katex.renderToString(tex, { output: "html", throwOnError: true });
    return { ok: true, glyphs: glyphs(html) };
  } catch (error) {
    return { ok: false, error: String(error.message).slice(0, 200) };
  }
});
process.stdout.write(JSON.stringify(results));
