package einox

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"

	einoopenai "github.com/cloudwego/eino-ext/components/model/openai"
	"github.com/cloudwego/eino/schema"
	"gopkg.in/yaml.v2"
)

// 直接使用原始结构体类型
type AzureCredential struct {
	Name         string   `yaml:"name"`
	ApiKey       string   `yaml:"api_key"`
	Endpoint     string   `yaml:"endpoint"`
	DeploymentId string   `yaml:"deployment_id"`
	ApiVersion   string   `yaml:"api_version"`
	Enabled      bool     `yaml:"enabled"`
	Weight       int      `yaml:"weight"`
	QPSLimit     int      `yaml:"qps_limit"`
	Description  string   `yaml:"description"`
	Models       []string `yaml:"models"`
	Timeout      int      `yaml:"timeout"`
	Proxy        string   `yaml:"proxy"`
}

// 修改配置文件结构定义
var azureConfig struct {
	Environments map[string]struct {
		Credentials []AzureCredential `yaml:"credentials"`
	} `yaml:"environments"`
}

// getAzureConfig 获取Azure配置
func (c *Config) getAzureConfig() (*einoopenai.ChatModelConfig, error) {
	// 使用统一定义的环境变量
	env := ENV
	if env == "" {
		env = "development"
	}

	//读取环境变量
	err := LoadLLMConfigPathFromEnv()
	if err != nil {
		return nil, fmt.Errorf("读取LLM配置路径失败: %v", err)
	}

	// 读取Azure配置文件
	yamlFile, err := os.ReadFile(filepath.Join(LLMConfigPath, "azure.yaml"))
	if err != nil {
		return nil, fmt.Errorf("读取Azure配置文件失败: %v", err)
	}

	err = yaml.Unmarshal(yamlFile, &azureConfig)
	if err != nil {
		fmt.Printf("解析Azure配置文件失败: %v", err)
		//抛出异常
		return nil, err
	}

	// 获取指定环境的配置
	envConfig, ok := azureConfig.Environments[env]
	if !ok {
		return nil, fmt.Errorf("未找到环境 %s 的配置", env)
	}

	// 存储启用的配置
	var enabledCredentials []AzureCredential

	// 遍历该环境下的所有凭证配置
	for _, cred := range envConfig.Credentials {
		// 只添加启用的配置
		if cred.Enabled {
			enabledCredentials = append(enabledCredentials, cred)
		}
	}

	// 如果没有启用的配置,返回错误
	if len(enabledCredentials) == 0 {
		return nil, fmt.Errorf("环境 %s 中没有启用的配置", env)
	}

	// 根据权重选择配置
	var selectedCred AzureCredential
	if len(enabledCredentials) > 1 {
		// 计算总权重
		totalWeight := 0
		for _, cred := range enabledCredentials {
			totalWeight += cred.Weight
		}

		// 生成一个随机数,范围是[0, totalWeight)
		randomNum := rand.Intn(totalWeight)

		// 根据权重选择配置
		currentWeight := 0

		for _, cred := range enabledCredentials {
			currentWeight += cred.Weight
			if randomNum < currentWeight {
				selectedCred = cred
				break
			}
		}
	} else {
		// 如果只有一个配置,直接使用
		selectedCred = enabledCredentials[0]
	}

	// 确保微软Azure配置存在
	if c.VendorOptional == nil {
		c.VendorOptional = &VendorOptional{}
	}
	if c.VendorOptional.AzureConfig == nil {
		c.VendorOptional.AzureConfig = &AzureConfig{}
	}

	//判断c.VendorOptional.AzureConfig.HTTPClient 可完善优化
	if c.VendorOptional.AzureConfig.HTTPClient == nil {
		c.VendorOptional.AzureConfig.HTTPClient = &http.Client{}
	}

	//判断代理设置不为空设置代理 可完善优化
	if selectedCred.Proxy != "" {
		c.VendorOptional.AzureConfig.HTTPClient.Transport = &http.Transport{
			Proxy: func(req *http.Request) (*url.URL, error) {
				return url.Parse(selectedCred.Proxy)
			},
		}
	}

	//selectedCred.Timeout大于0时设置请求超时时间
	if selectedCred.Timeout > 0 {
		c.VendorOptional.AzureConfig.HTTPClient.Timeout = time.Duration(selectedCred.Timeout) * time.Second
	}

	//selectedCred.ApiKey 解密
	// 第一次初始化，应该生成新的密钥文件
	_, decryptFunc1, err := InitRSAKeyManager()
	if err != nil {
		return nil, fmt.Errorf("初始化RSA密钥管理器失败: %v", err)
	}
	decryptedApiKey, err := decryptFunc1(selectedCred.ApiKey)
	if err != nil {
		return nil, fmt.Errorf("解密失败: %v", err)
	}
	selectedCred.ApiKey = decryptedApiKey // 更新为解密后的 key

	nConf := &einoopenai.ChatModelConfig{
		ByAzure:     true,
		APIKey:      selectedCred.ApiKey,
		BaseURL:     selectedCred.Endpoint,
		APIVersion:  selectedCred.ApiVersion,
		Model:       c.Model,
		MaxTokens:   &c.MaxTokens,
		Temperature: c.Temperature,
		TopP:        c.TopP,
		Stop:        c.Stop,
		// 补充额外参数
		HTTPClient:       c.VendorOptional.AzureConfig.HTTPClient,
		PresencePenalty:  c.VendorOptional.AzureConfig.PresencePenalty,
		FrequencyPenalty: c.VendorOptional.AzureConfig.FrequencyPenalty,
		LogitBias:        c.VendorOptional.AzureConfig.LogitBias,
		ResponseFormat:   c.VendorOptional.AzureConfig.ResponseFormat,
		Seed:             c.VendorOptional.AzureConfig.Seed,
		User:             c.VendorOptional.AzureConfig.User,
	}
	return nConf, nil
}

// convertOpenAIToolsToSchemaTools 将 openai.Tool 转换为 schema.ToolInfo
func convertOpenAIToolsToSchemaTools(tools []openai.Tool) ([]*schema.ToolInfo, error) {
	if tools == nil {
		return nil, nil
	}

	schemaTools := make([]*schema.ToolInfo, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != openai.ToolTypeFunction || tool.Function == nil {
			continue
		}

		// 创建一个基本的 ToolInfo 实例
		schemaTool := &schema.ToolInfo{
			Name: tool.Function.Name,
			Desc: tool.Function.Description,
			// 转换参数定义 (这部分需要根据实际情况实现或假设 einoopenai 会处理)
			// Assuming parameters are handled downstream or this is sufficient for now
			// ParamsOneOf: ???
		}

		schemaTools = append(schemaTools, schemaTool)
	}
	return schemaTools, nil
}

// convertSchemaToolCallsToOpenAI 将 schema.ToolCall 转换为 openai.ToolCall
func convertSchemaToolCallsToOpenAI(schemaCalls []schema.ToolCall) []openai.ToolCall {
	if schemaCalls == nil || len(schemaCalls) == 0 {
		return nil
	}

	openAICalls := make([]openai.ToolCall, 0, len(schemaCalls))
	for _, sc := range schemaCalls {
		// 默认使用 function 类型，Eino 中默认也是 "function"
		toolType := openai.ToolTypeFunction
		if sc.Type != "function" {
			fmt.Printf("Warning: Unknown tool type '%s' for ToolCall ID %s, defaulting to 'function'\n", sc.Type, sc.ID)
		}

		openAICalls = append(openAICalls, openai.ToolCall{
			ID:   sc.ID,
			Type: toolType,
			Function: openai.FunctionCall{
				Name:      sc.Function.Name,
				Arguments: sc.Function.Arguments,
			},
		})
	}
	return openAICalls
}

// convertSchemaStreamToolCallsToOpenAI 将 schema.ToolCall 转换为流式 openai.ToolCall
func convertSchemaStreamToolCallsToOpenAI(schemaCalls []schema.ToolCall) []openai.ToolCall {
	if schemaCalls == nil || len(schemaCalls) == 0 {
		return nil
	}

	openAICalls := make([]openai.ToolCall, 0, len(schemaCalls))
	for _, sc := range schemaCalls {
		var localIndex int
		if sc.Index != nil {
			localIndex = *sc.Index
		} else {
			// 如果 Index 为 nil，可能需要设置一个默认值或根据上下文决定
			// 对于流式响应，通常需要索引，这里暂时设为 0，可能需要调整
			localIndex = 0
			fmt.Printf("Warning: Stream tool call index is nil for ID %s, defaulting to 0\n", sc.ID)
		}

		toolType := openai.ToolTypeFunction
		if sc.Type != "function" {
			fmt.Printf("Warning: Unknown stream tool type '%s' for ToolCall ID %s, defaulting to 'function'\n", sc.Type, sc.ID)
		}

		openAICalls = append(openAICalls, openai.ToolCall{
			Index: &localIndex,
			ID:    sc.ID,
			Type:  toolType,
			Function: openai.FunctionCall{
				Name:      sc.Function.Name,
				Arguments: sc.Function.Arguments,
			},
		})
	}
	return openAICalls
}

// AzureCreateChatCompletion 使用Azure OpenAI服务创建聊天完成
func AzureCreateChatCompletion(req ChatRequest) (*openai.ChatCompletionResponse, error) {
	// 创建Azure OpenAI配置
	conf := &Config{
		Vendor:      "azure",
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: &req.Temperature,
		TopP:        &req.TopP,
		Stop:        req.Stop,
	}

	// 获取Azure配置
	azureConf, err := conf.getAzureConfig()
	if err != nil {
		return nil, fmt.Errorf("获取Azure配置失败: %v", err)
	}
	azureConf.Model = req.Model // 将请求中的模型设置到配置中

	// 创建上下文
	ctx := context.Background()

	// 创建聊天模型
	chatModel, err := einoopenai.NewChatModel(ctx, azureConf)
	if err != nil {
		return nil, fmt.Errorf("创建聊天模型失败: %v", err)
	}

	// --- 工具绑定逻辑 ---
	hasTools := len(req.ChatCompletionRequest.Tools) > 0
	if hasTools {
		// 将OpenAI工具格式转换为Eino的工具格式
		schemaTools, err := convertOpenAIToolsToSchemaTools(req.ChatCompletionRequest.Tools)
		if err != nil {
			return nil, fmt.Errorf("转换工具定义失败: %w", err)
		}

		if len(schemaTools) > 0 {
			// 根据toolChoice决定是否使用强制工具绑定
			toolChoiceStr := fmt.Sprintf("%v", req.ChatCompletionRequest.ToolChoice) // 确保是字符串
			if strings.ToLower(toolChoiceStr) == "required" || strings.ToLower(toolChoiceStr) == "force" {
				err = chatModel.BindForcedTools(schemaTools) // 使用强制工具绑定
				if err != nil {
					return nil, fmt.Errorf("强制绑定工具失败: %w", err)
				}
			} else {
				err = chatModel.BindTools(schemaTools) // 使用常规工具绑定 (auto or specific tool)
				if err != nil {
					return nil, fmt.Errorf("绑定工具失败: %w", err)
				}
			}
		}
	}
	// --- 工具绑定逻辑结束 ---

	// 转换消息格式，使用通用方法
	schemaMessages := convertChatRequestToSchemaMessages(req)

	// 调用Generate方法获取响应
	resp, err := chatModel.Generate(ctx, schemaMessages)
	if err != nil {
		// 尝试解析 Azure 特定的错误信息
		var apiError *openai.APIError
		if errors.As(err, &apiError) {
			return nil, fmt.Errorf("调用Generate方法失败 (Azure API Error: Status=%d Type=%s Code=%v Param=%v): %w",
				apiError.HTTPStatusCode, apiError.Type, apiError.Code, apiError.Param, err)
		}
		return nil, fmt.Errorf("调用Generate方法失败: %v", err)
	}

	// --- 处理工具调用响应 ---
	choices := []openai.ChatCompletionChoice{
		{
			Index: 0,
			Message: openai.ChatCompletionMessage{
				Role:    string(resp.Role),
				Content: resp.Content,
				// 检查 resp 是否包含工具调用信息并进行转换
				ToolCalls: convertSchemaToolCallsToOpenAI(resp.ToolCalls),
			},
			FinishReason: openai.FinishReason(resp.ResponseMeta.FinishReason),
		},
	}
	// --- 工具调用响应处理结束 ---

	// 生成唯一ID
	uniqueID := fmt.Sprintf("azure-%d", time.Now().UnixNano())

	// 获取Token使用情况
	var usage openai.Usage
	if resp.ResponseMeta != nil && resp.ResponseMeta.Usage != nil {
		usage = openai.Usage{
			PromptTokens:     resp.ResponseMeta.Usage.PromptTokens,
			CompletionTokens: resp.ResponseMeta.Usage.CompletionTokens,
			TotalTokens:      resp.ResponseMeta.Usage.TotalTokens,
		}
	} else {
		fmt.Println("Info: No usage info found in response meta")
	}

	// 构造并返回响应
	finalResp := &openai.ChatCompletionResponse{
		ID:      uniqueID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model, // 使用请求中的模型名称
		Choices: choices,   // 使用上面构造的 choices
		Usage:   usage,
	}
	return finalResp, nil
}

// 检查消息中是否包含工具消息
func containsToolMessages(messages []openai.ChatCompletionMessage) bool {
	for _, msg := range messages {
		if msg.Role == openai.ChatMessageRoleTool || msg.ToolCallID != "" {
			return true
		}
		if len(msg.ToolCalls) > 0 {
			return true
		}
	}
	return false
}

// AzureCreateChatCompletionToChat 使用Azure OpenAI服务创建聊天完成接口
func AzureCreateChatCompletionToChat(req ChatRequest) (*openai.ChatCompletionResponse, error) {
	// 准备请求参数
	model := req.Model
	if model == "" {
		return nil, fmt.Errorf("未指定模型名称")
	}

	// 调用Azure服务 (现在会处理工具调用)
	resp, err := AzureCreateChatCompletion(req)
	if err != nil {
		// 错误信息已在 AzureCreateChatCompletion 中格式化和记录
		return nil, fmt.Errorf("调用Azure聊天接口失败: %w", err)
	}

	return resp, nil
}

// AzureStreamChatCompletion 使用Azure OpenAI服务创建流式聊天完成
func AzureStreamChatCompletion(req ChatRequest) (*schema.StreamReader[*openai.ChatCompletionStreamResponse], error) {
	// 创建Azure OpenAI配置
	conf := &Config{
		Vendor:      "azure",
		Model:       req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: &req.Temperature,
		TopP:        &req.TopP,
		Stop:        req.Stop,
	}

	// 获取Azure配置
	azureConf, err := conf.getAzureConfig()
	if err != nil {
		return nil, fmt.Errorf("获取Azure配置失败: %v", err)
	}
	azureConf.Model = req.Model // 确保使用请求中的模型

	// 创建上下文
	ctx := context.Background()

	// 创建聊天模型
	chatModel, err := einoopenai.NewChatModel(ctx, azureConf)
	if err != nil {
		return nil, fmt.Errorf("创建聊天模型失败: %v", err)
	}

	// --- 添加工具绑定逻辑 ---
	hasTools := len(req.ChatCompletionRequest.Tools) > 0
	if hasTools {
		schemaTools, err := convertOpenAIToolsToSchemaTools(req.ChatCompletionRequest.Tools)
		if err != nil {
			return nil, fmt.Errorf("转换工具定义失败: %w", err)
		}
		if len(schemaTools) > 0 {
			err = chatModel.BindTools(schemaTools) // 调用 BindTools
			if err != nil {
				return nil, fmt.Errorf("绑定工具失败: %w", err)
			}
		}
	}
	// --- 工具绑定逻辑结束 ---

	// 转换消息格式，使用通用方法
	schemaMessages := convertChatRequestToSchemaMessages(req)

	// 调用Stream方法获取流式响应
	streamReader, err := chatModel.Stream(ctx, schemaMessages)
	if err != nil {
		return nil, fmt.Errorf("调用Stream方法失败: %v", err)
	}

	// 创建结果通道
	resultReader, resultWriter := schema.Pipe[*openai.ChatCompletionStreamResponse](10)

	// 启动goroutine处理流式数据
	go func() {
		defer func() {
			if panicErr := recover(); panicErr != nil {
				fmt.Printf("Panic recovered in stream processing goroutine: %v\n", panicErr)
			}
			streamReader.Close()
			resultWriter.Close()
		}()

		// 生成唯一ID
		uniqueID := fmt.Sprintf("azure-stream-%d", time.Now().UnixNano())
		created := time.Now().Unix()

		for {
			// 从流中接收消息
			message, err := streamReader.Recv()
			if errors.Is(err, io.EOF) {
				break // 流结束
			}
			if err != nil {
				_ = resultWriter.Send(nil, fmt.Errorf("从Azure接收流数据失败: %v", err))
				return
			}

			// 构造流式响应
			streamResp := &openai.ChatCompletionStreamResponse{
				ID:      uniqueID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   req.Model, // 使用请求中的模型
				Choices: []openai.ChatCompletionStreamChoice{
					{
						Index: 0,
						Delta: openai.ChatCompletionStreamChoiceDelta{
							Role:    string(message.Role), // Role 可能为空或 "assistant"
							Content: message.Content,
							// 检查 message 是否包含工具调用信息并进行转换
							ToolCalls: convertSchemaStreamToolCallsToOpenAI(message.ToolCalls),
						},
						FinishReason: "", // 在最后一条消息中设置
					},
				},
				// 流式响应通常不包含 Usage
			}

			// 处理 FinishReason
			if message.ResponseMeta != nil && message.ResponseMeta.FinishReason != "" {
				finishReason := openai.FinishReason(message.ResponseMeta.FinishReason)
				streamResp.Choices[0].FinishReason = finishReason
			}

			// 发送流式响应
			closed := resultWriter.Send(streamResp, nil)
			if closed {
				return
			}
		}
	}()

	return resultReader, nil
}

// --- 添加辅助函数 ---

// AzureStreamChatCompletionToChat 使用Azure OpenAI服务创建流式聊天完成并转换为聊天流格式
func AzureStreamChatCompletionToChat(req ChatRequest, writer io.Writer) error {
	// 调用Azure流式聊天API (现在会处理工具)
	streamReader, err := AzureStreamChatCompletion(req)
	if err != nil {
		return fmt.Errorf("调用Azure流式聊天接口失败: %w", err)
	}

	// 处理流式响应
	chunkCount := 0
	for {
		response, err := streamReader.Recv() // response 是 *openai.ChatCompletionStreamResponse
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			// 将错误写入流中，如果 writer 支持的话，或者直接返回错误
			return fmt.Errorf("接收Azure流式响应失败: %w", err)
		}

		// response 已经是 *openai.ChatCompletionStreamResponse 类型，直接序列化
		if response == nil {
			fmt.Println("Info: Received nil response from streamReader, skipping.")
			continue
		}
		chunkCount++

		// 将响应写入writer
		data, err := json.Marshal(response)
		if err != nil {
			// 记录错误，但尝试继续处理流
			fmt.Printf("序列化流式响应失败: %v\n", err)
			continue
		}

		// 添加data:前缀
		if _, err := writer.Write([]byte("data: ")); err != nil {
			return fmt.Errorf("写入流式响应前缀失败: %w", err)
		}

		if _, err := writer.Write(data); err != nil {
			return fmt.Errorf("写入流式响应失败: %w", err)
		}

		if _, err := writer.Write([]byte("\n\n")); err != nil {
			return fmt.Errorf("写入流式响应分隔符失败: %w", err)
		}
	}

	// 添加结束标记
	if _, err := writer.Write([]byte("data: [DONE]\n\n")); err != nil {
		return fmt.Errorf("写入流式响应结束标记失败: %w", err)
	}

	return nil
}
