import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { DuckDuckGoSearch } from "@langchain/community/tools/duckduckgo_search";
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import { pull } from "langchain/hub";
import {
  AgentExecutor,
  createReactAgent,
  createOpenAIFunctionsAgent,
} from "langchain/agents";

const llm = new ChatOpenAI({
  model: "gpt-3.5-turbo",
  temperature: 0,
  apiKey: process.env.OPENAI_API_KEY,
});

const tools = [
  // https://js.langchain.com/v0.2/docs/integrations/tools/duckduckgo_search/
  new DuckDuckGoSearch({ maxResults: 1 }),
  // https://js.langchain.com/v0.2/docs/integrations/tools/wikipedia/
  new WikipediaQueryRun({ topKResults: 3, maxDocContentLength: 4000 }),
];

const prompt = await pull("hwchase17/openai-functions-agent");
const agent = await createOpenAIFunctionsAgent({
  llm,
  tools,
  prompt,
});

// const prompt = await pull("hwchase17/react");
// const agent = await createReactAgent({
//   llm,
//   tools,
//   prompt,
// });

const agentExecutor = new AgentExecutor({
  agent,
  tools,
  returnIntermediateSteps: true,
});

const input =
  "Vou viajar para Londres em Agosto de 2024. Faça para mim um roteiro de viagem para mim com eventos que irão ocorrer na data da viagem e com o preço de passagem de São Paulo para Londres.";

const result = await agentExecutor.invoke({
  input,
});

console.log("------");
console.log(result.output);
console.log("------");
console.log(result.intermediateSteps);
