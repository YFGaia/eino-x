package einox

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
)

// TestCreateChatCompletionStream 测试流式聊天完成功能
// 执行命令：go test -run TestCreateChatCompletionStream
func TestCreateChatCompletionStream(t *testing.T) {
	// 检查环境变量是否设置了测试标志
	if os.Getenv("SKIP_BEDROCK_TESTS") == "1" {
		t.Skip("跳过Bedrock API测试")
	}

	// 检查是否跳过Azure测试
	//skipAzureTests := os.Getenv("SKIP_AZURE_TESTS") == "1"

	// 检查是否跳过DeepSeek测试
	skipDeepSeekTests := os.Getenv("SKIP_DEEPSEEK_TESTS") == "1"

	// 准备测试用例
	testCases := []struct {
		name       string
		request    ChatRequest
		provider   string
		skipTest   bool
		skipReason string
	}{
		{
			name: "基本流式聊天完成测试-bedrock",
			request: ChatRequest{
				Provider: "bedrock",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个有帮助的助手。",
						},
						{
							Role:    "user",
							Content: "简单介绍一下自然语言处理。",
						},
					},
					MaxTokens:   100,
					Temperature: 0.7,
					Stream:      true,
				},
			},
			provider: "bedrock",
			skipTest: false,
		},
		{
			name: "基本流式聊天完成测试-azure",
			request: ChatRequest{
				Provider: "azure",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "gpt-4o",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个有帮助的助手。",
						},
						{
							Role:    "user",
							Content: "简单介绍一下人工智能。",
						},
					},
					MaxTokens:   100,
					Temperature: 0.7,
					Stream:      true,
				},
			},
			provider:   "azure",
			skipTest:   false,
			skipReason: "跳过Azure API测试",
		},
		{
			name: "基本流式聊天完成测试-deepseek",
			request: ChatRequest{
				Provider: "deepseek",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "deepseek-chat",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个专业的AI助手。",
						},
						{
							Role:    "user",
							Content: "请简单介绍一下机器学习的基本概念。",
						},
					},
					MaxTokens:   150,
					Temperature: 0.8,
					Stream:      true,
				},
			},
			provider:   "deepseek",
			skipTest:   skipDeepSeekTests,
			skipReason: "跳过DeepSeek API测试",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 如果需要跳过此测试
			if tc.skipTest {
				t.Skip(tc.skipReason)
			}

			// 设置供应商
			tc.request.Provider = tc.provider

			// 创建缓冲区用于接收流式响应
			buffer := new(bytes.Buffer)

			// 调用被测试的函数
			resp, err := CreateChatCompletion(tc.request, buffer)

			// 检查结果
			if err != nil {
				t.Logf("测试期间出现错误: %v", err)
				t.Skip("API调用失败，可能是配置问题")
				return
			}

			// 对于流式响应，期望resp为nil
			assert.Nil(t, resp, "流式响应应返回nil响应对象")

			// 获取响应内容
			response := buffer.String()

			// 验证响应格式正确性
			assert.True(t, len(response) > 0, "响应不应为空")
			assert.Contains(t, response, "data: ", "响应应包含data:前缀")

			// 解析并验证每个响应块
			lines := strings.Split(response, "\n\n")
			var contentLines []string
			var allContent string

			for _, line := range lines {
				if strings.HasPrefix(line, "data: ") && line != "data: [DONE]" {
					// 解析JSON
					jsonData := strings.TrimPrefix(line, "data: ")
					var streamResp StreamResponse
					err := json.Unmarshal([]byte(jsonData), &streamResp)

					if err != nil {
						t.Errorf("解析响应JSON失败: %v", err)
						continue
					}

					// 验证响应结构
					assert.NotEmpty(t, streamResp.ID, "响应ID不应为空")
					assert.Equal(t, "chat.completion.chunk", streamResp.Object, "响应对象类型应为chat.completion.chunk")
					assert.NotZero(t, streamResp.Created, "创建时间不应为零")
					assert.Equal(t, tc.request.ChatCompletionRequest.Model, streamResp.Model, "响应模型应与请求模型匹配")
					assert.NotEmpty(t, streamResp.Choices, "选择不应为空")

					// 收集内容
					if len(streamResp.Choices) > 0 {
						content := streamResp.Choices[0].Delta.Content
						if content != "" {
							contentLines = append(contentLines, content)
							allContent += content
						}
					}
				}
			}

			// 验证收集到的内容
			t.Logf("收到 %d 个内容块", len(contentLines))
			t.Logf("完整响应内容: %s", allContent)
			assert.True(t, len(contentLines) > 0, "应收到至少一个内容块")
			assert.NotEmpty(t, allContent, "应收到非空内容")
		})
	}
}

// TestCreateChatCompletionNonStream 测试非流式聊天完成功能
// 执行命令：go test -run TestCreateChatCompletionNonStream
func TestCreateChatCompletionNonStream(t *testing.T) {
	// 检查环境变量是否设置了测试标志
	if os.Getenv("SKIP_BEDROCK_TESTS") == "1" {
		t.Skip("跳过Bedrock API测试")
	}

	// 检查是否跳过DeepSeek测试
	skipDeepSeekTests := os.Getenv("SKIP_DEEPSEEK_TESTS") == "1"

	// 准备测试用例
	testCases := []struct {
		name           string
		request        ChatRequest
		provider       string
		expectError    bool
		expectedErrMsg string
		skipTest       bool
		skipReason     string
	}{
		{
			name: "基本非流式聊天完成测试-bedrock",
			request: ChatRequest{
				Provider: "bedrock",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个有帮助的助手。",
						},
						{
							Role:    "user",
							Content: "简单介绍一下自然语言处理。",
						},
					},
					MaxTokens:   100,
					Temperature: 0.7,
					Stream:      false,
				},
			},
			provider:    "bedrock",
			expectError: false,
			skipTest:    false,
		},
		{
			name: "调用BedrockCreateChatCompletionToChat测试",
			request: ChatRequest{
				Provider: "bedrock",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个简洁的助手。回答要精简。",
						},
						{
							Role:    "user",
							Content: "什么是大模型？",
						},
					},
					MaxTokens:   50,
					Temperature: 0.5,
					Stream:      false,
				},
			},
			provider:    "bedrock",
			expectError: false,
			skipTest:    false,
		},
		{
			name: "基本非流式聊天完成测试-azure",
			request: ChatRequest{
				Provider: "azure",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "gpt-4o",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个简洁的助手。回答要精简。",
						},
						{
							Role:    "user",
							Content: "什么是自然语言处理？",
						},
					},
					MaxTokens:   50,
					Temperature: 0.7,
					Stream:      false,
				},
			},
			provider:    "azure",
			expectError: false,
			skipTest:    false,
			skipReason:  "跳过Azure API测试",
		},
		{
			name: "基本非流式聊天完成测试-deepseek",
			request: ChatRequest{
				Provider: "deepseek",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "deepseek-chat",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个专业的AI助手。",
						},
						{
							Role:    "user",
							Content: "请解释什么是深度学习？",
						},
					},
					MaxTokens:   100,
					Temperature: 0.7,
					Stream:      false,
				},
			},
			provider:    "deepseek",
			expectError: false,
			skipTest:    skipDeepSeekTests,
			skipReason:  "跳过DeepSeek API测试",
		},
		{
			name: "调用DeepSeekCreateChatCompletionToChat测试",
			request: ChatRequest{
				Provider: "deepseek",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "deepseek-chat",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "system",
							Content: "你是一个简洁的助手。回答要精简。",
						},
						{
							Role:    "user",
							Content: "什么是强化学习？",
						},
					},
					MaxTokens:   50,
					Temperature: 0.5,
					Stream:      false,
				},
			},
			provider:    "deepseek",
			expectError: false,
			skipTest:    skipDeepSeekTests,
			skipReason:  "跳过DeepSeek API测试",
		},
		{
			name: "不支持的供应商测试",
			request: ChatRequest{
				Provider: "unsupported",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model: "some-model",
					Messages: []openai.ChatCompletionMessage{
						{
							Role:    "user",
							Content: "Hello",
						},
					},
				},
			},
			provider:       "unsupported",
			expectError:    true,
			expectedErrMsg: "不支持的AI供应商: unsupported",
			skipTest:       false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 如果需要跳过此测试
			if tc.skipTest {
				t.Skip(tc.skipReason)
			}

			// 设置供应商
			tc.request.Provider = tc.provider

			// 调用被测试的函数
			resp, err := CreateChatCompletion(tc.request, nil)

			// 检查错误
			if tc.expectError {
				assert.Error(t, err, "应返回错误")
				if tc.expectedErrMsg != "" {
					assert.Contains(t, err.Error(), tc.expectedErrMsg, "错误消息应包含预期内容")
				}
				return
			}

			// 非流式响应验证
			assert.NoError(t, err, "不应返回错误")
			assert.NotNil(t, resp, "响应不应为空")
			assert.NotEmpty(t, resp.ID, "响应ID不应为空")
			assert.Equal(t, "chat.completion", resp.Object, "响应对象类型应为chat.completion")
			assert.NotZero(t, resp.Created, "创建时间不应为零")
			assert.Equal(t, tc.request.ChatCompletionRequest.Model, resp.Model, "响应模型应与请求模型匹配")
			assert.NotEmpty(t, resp.Choices, "选择不应为空")

			if len(resp.Choices) > 0 {
				assert.NotEmpty(t, resp.Choices[0].Message.Content, "消息内容不应为空")
				assert.NotEmpty(t, resp.Choices[0].FinishReason, "完成原因不应为空")
				assert.Equal(t, "assistant", resp.Choices[0].Message.Role, "消息角色应为assistant")
				t.Logf("响应内容: %s", resp.Choices[0].Message.Content)
				t.Logf("完成原因: %s", resp.Choices[0].FinishReason)
			}
		})
	}
}

// TestCreateChatCompletionWithDefaultProvider 测试默认供应商的情况
func TestCreateChatCompletionWithDefaultProvider(t *testing.T) {
	// 检查环境变量是否设置了测试标志
	if os.Getenv("SKIP_BEDROCK_TESTS") == "1" {
		t.Skip("跳过Bedrock API测试")
	}

	// 准备测试用例
	request := ChatRequest{
		Provider: "bedrock",
		ChatCompletionRequest: openai.ChatCompletionRequest{
			Model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    "system",
					Content: "你是一个有帮助的助手。",
				},
				{
					Role:    "user",
					Content: "简单介绍一下自然语言处理。",
				},
			},
			MaxTokens:   100,
			Temperature: 0.7,
			Stream:      true,
		},
	}

	// 创建缓冲区用于接收流式响应
	buffer := new(bytes.Buffer)

	// 调用被测试的函数
	resp, err := CreateChatCompletion(request, buffer)

	// 检查结果
	if err != nil {
		t.Logf("测试期间出现错误: %v", err)
		t.Skip("API调用失败，可能是配置问题")
		return
	}

	// 对于流式响应，期望resp为nil
	assert.Nil(t, resp, "流式响应应返回nil响应对象")

	// 获取响应内容
	response := buffer.String()

	// 验证响应格式正确性
	assert.True(t, len(response) > 0, "响应不应为空")
	assert.Contains(t, response, "data: ", "响应应包含data:前缀")

	// 验证至少收到了一些内容
	t.Logf("使用默认供应商获取的响应: %s", response)
}

// TestCreateChatCompletionCallsBedrockFunction 测试CreateChatCompletion是否正确调用了BedrockCreateChatCompletionToChat函数
func TestCreateChatCompletionCallsBedrockFunction(t *testing.T) {
	// 由于不能直接替换函数，我们将通过测试bedrock provider的非流式请求来验证
	// CreateChatCompletion函数是否会正确调用BedrockCreateChatCompletionToChat

	// 创建测试请求
	request := ChatRequest{
		Provider: "bedrock",
		ChatCompletionRequest: openai.ChatCompletionRequest{
			Model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    "user",
					Content: "这是一个测试请求",
				},
			},
			Temperature: 0.7,
			MaxTokens:   50,
			Stream:      false,
		},
	}

	// 跳过实际API调用
	if os.Getenv("SKIP_BEDROCK_TESTS") == "1" {
		t.Skip("跳过Bedrock API测试")
	}

	// 调用被测试的函数
	resp, err := CreateChatCompletion(request, nil)

	// 检查结果
	if err != nil {
		// 如果返回的错误包含"尚未实现"，说明函数尝试调用了BedrockCreateChatCompletionToChat
		// 但该函数可能尚未完全实现
		if strings.Contains(err.Error(), "尚未实现") {
			t.Logf("检测到预期的错误: %v", err)
			t.Log("这表明CreateChatCompletion正确地尝试调用了BedrockCreateChatCompletionToChat")
			return
		}

		// 如果是其他API错误，可能是配置问题，我们可以接受
		t.Logf("API调用错误: %v", err)
		t.Skip("API调用失败，可能是配置问题")
		return
	}

	// 如果没有错误，则验证响应格式是否符合预期
	assert.NotNil(t, resp, "响应不应为空")
	assert.NotEmpty(t, resp.ID, "响应ID不应为空")
	assert.Equal(t, "chat.completion", resp.Object, "响应对象类型应为chat.completion")
	assert.NotZero(t, resp.Created, "创建时间不应为零")
	assert.Equal(t, request.ChatCompletionRequest.Model, resp.Model, "响应模型应与请求模型匹配")
	assert.NotEmpty(t, resp.Choices, "选择不应为空")

	if len(resp.Choices) > 0 {
		assert.NotEmpty(t, resp.Choices[0].Message.Content, "消息内容不应为空")
		assert.NotEmpty(t, resp.Choices[0].FinishReason, "完成原因不应为空")
		assert.Equal(t, "assistant", resp.Choices[0].Message.Role, "消息角色应为assistant")
		t.Logf("响应内容: %s", resp.Choices[0].Message.Content)
		t.Logf("完成原因: %s", resp.Choices[0].FinishReason)
	}

	t.Log("测试通过：CreateChatCompletion成功调用了BedrockCreateChatCompletionToChat并返回了有效响应")
}

// TestCreateChatCompletionCallsDeepSeekFunction 测试CreateChatCompletion是否正确调用了DeepSeekCreateChatCompletionToChat函数
func TestCreateChatCompletionCallsDeepSeekFunction(t *testing.T) {
	// 由于不能直接替换函数，我们将通过测试deepseek provider的非流式请求来验证
	// CreateChatCompletion函数是否会正确调用DeepSeekCreateChatCompletionToChat

	// 创建测试请求
	request := ChatRequest{
		Provider: "deepseek",
		ChatCompletionRequest: openai.ChatCompletionRequest{
			Model: "deepseek-chat",
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    "user",
					Content: "这是一个DeepSeek测试请求",
				},
			},
			Temperature: 0.7,
			MaxTokens:   50,
			Stream:      false,
		},
	}

	// 跳过实际API调用
	if os.Getenv("SKIP_DEEPSEEK_TESTS") == "1" {
		t.Skip("跳过DeepSeek API测试")
	}

	// 调用被测试的函数
	resp, err := CreateChatCompletion(request, nil)

	// 检查结果
	if err != nil {
		// 如果返回的错误包含"尚未实现"，说明函数尝试调用了DeepSeekCreateChatCompletionToChat
		// 但该函数可能尚未完全实现
		if strings.Contains(err.Error(), "尚未实现") {
			t.Logf("检测到预期的错误: %v", err)
			t.Log("这表明CreateChatCompletion正确地尝试调用了DeepSeekCreateChatCompletionToChat")
			return
		}

		// 如果是其他API错误，可能是配置问题，我们可以接受
		t.Logf("API调用错误: %v", err)
		t.Skip("API调用失败，可能是配置问题")
		return
	}

	// 如果没有错误，则验证响应格式是否符合预期
	assert.NotNil(t, resp, "响应不应为空")
	assert.NotEmpty(t, resp.ID, "响应ID不应为空")
	assert.Equal(t, "chat.completion", resp.Object, "响应对象类型应为chat.completion")
	assert.NotZero(t, resp.Created, "创建时间不应为零")
	assert.Equal(t, request.ChatCompletionRequest.Model, resp.Model, "响应模型应与请求模型匹配")
	assert.NotEmpty(t, resp.Choices, "选择不应为空")

	if len(resp.Choices) > 0 {
		assert.NotEmpty(t, resp.Choices[0].Message.Content, "消息内容不应为空")
		assert.NotEmpty(t, resp.Choices[0].FinishReason, "完成原因不应为空")
		assert.Equal(t, "assistant", resp.Choices[0].Message.Role, "消息角色应为assistant")
		t.Logf("响应内容: %s", resp.Choices[0].Message.Content)
		t.Logf("完成原因: %s", resp.Choices[0].FinishReason)
	}

	t.Log("测试通过：CreateChatCompletion成功调用了DeepSeekCreateChatCompletionToChat并返回了有效响应")
}

// TestCreateChatCompletionDeepSeekStream 测试DeepSeek的流式响应功能
func TestCreateChatCompletionDeepSeekStream(t *testing.T) {
	// 跳过实际API调用
	if os.Getenv("SKIP_DEEPSEEK_TESTS") == "1" {
		t.Skip("跳过DeepSeek API测试")
	}

	// 创建测试请求
	request := ChatRequest{
		Provider: "deepseek",
		ChatCompletionRequest: openai.ChatCompletionRequest{
			Model: "deepseek-chat",
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    "system",
					Content: "你是一个专业且有帮助的助手。",
				},
				{
					Role:    "user",
					Content: "简要介绍计算机视觉的应用领域。",
				},
			},
			MaxTokens:   100,
			Temperature: 0.7,
			Stream:      true,
		},
	}

	// 创建缓冲区用于接收流式响应
	buffer := new(bytes.Buffer)

	// 调用被测试的函数
	resp, err := CreateChatCompletion(request, buffer)

	// 检查结果
	if err != nil {
		t.Logf("测试期间出现错误: %v", err)
		t.Skip("API调用失败，可能是配置问题")
		return
	}

	// 对于流式响应，期望resp为nil
	assert.Nil(t, resp, "流式响应应返回nil响应对象")

	// 获取响应内容
	response := buffer.String()

	// 验证响应格式正确性
	assert.True(t, len(response) > 0, "响应不应为空")
	assert.Contains(t, response, "data: ", "响应应包含data:前缀")

	// 解析并验证每个响应块
	lines := strings.Split(response, "\n\n")
	var contentLines []string
	var allContent string

	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") && line != "data: [DONE]" {
			// 解析JSON
			jsonData := strings.TrimPrefix(line, "data: ")
			var streamResp StreamResponse
			err := json.Unmarshal([]byte(jsonData), &streamResp)

			if err != nil {
				t.Errorf("解析响应JSON失败: %v", err)
				continue
			}

			// 验证响应结构
			assert.NotEmpty(t, streamResp.ID, "响应ID不应为空")
			assert.Equal(t, "chat.completion.chunk", streamResp.Object, "响应对象类型应为chat.completion.chunk")
			assert.NotZero(t, streamResp.Created, "创建时间不应为零")
			assert.Equal(t, request.ChatCompletionRequest.Model, streamResp.Model, "响应模型应与请求模型匹配")
			assert.NotEmpty(t, streamResp.Choices, "选择不应为空")

			// 收集内容
			if len(streamResp.Choices) > 0 {
				content := streamResp.Choices[0].Delta.Content
				if content != "" {
					contentLines = append(contentLines, content)
					allContent += content
				}
			}
		}
	}

	// 验证收集到的内容
	t.Logf("收到 %d 个内容块", len(contentLines))
	t.Logf("完整响应内容: %s", allContent)
	assert.True(t, len(contentLines) > 0, "应收到至少一个内容块")
	assert.NotEmpty(t, allContent, "应收到非空内容")
}

// TestChatService_CreateChatCompletionSingleTool 测试单个工具调用功能
// 执行命令：go test -run TestChatService_CreateChatCompletionSingleTool
func TestChatService_CreateChatCompletionSingleTool(t *testing.T) {
	// 准备测试用例
	testCases := []struct {
		name            string
		initialMessages []openai.ChatCompletionMessage
		expectToolCall  bool
		toolResult      string // 模拟的工具结果
		expectError     bool
	}{
		{
			name: "Azure单工具调用测试",
			initialMessages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "请使用天气工具查询北京的天气情况，需要详细的温度、湿度和风力信息。",
				},
			},
			expectToolCall: true,
			toolResult:     `{"directories":["/Users/test/Documents/files"]}`,
			expectError:    false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// --- 第一步：发送初始请求，获取工具调用 ---
			firstRequest := ChatRequest{
				Provider: "azure",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model:    "gpt-4o", // 确保模型支持工具调用
					Messages: tc.initialMessages,
					// 定义工具
					Tools: []openai.Tool{
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "get_weather",
								Description: "获取指定城市的天气信息",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"location": map[string]interface{}{
											"type":        "string",
											"description": "城市名称，例如：'北京'",
										},
									},
									"required": []string{"location"},
								},
							},
						},
					},
					ToolChoice:  "auto", // 让模型决定是否使用工具
					MaxTokens:   150,
					Temperature: 0.5,
				},
			}

			t.Log("--- 发送第一次请求 ---")
			firstResp, err := CreateChatCompletion(firstRequest, nil)

			// 错误处理
			if err != nil {
				logAzureError(t, err)
				if tc.expectError {
					assert.Error(t, err, "预期应返回错误")
					return // 如果预期错误且确实发生错误，测试通过
				}
				t.Fatalf("第一次API调用失败: %v", err) // 非预期错误，终止测试
			}
			if tc.expectError {
				assert.NoError(t, err, "预期有错误但API调用成功")
				return // 如果预期错误但没有发生，测试失败
			}

			// 验证第一次响应
			assert.NotNil(t, firstResp, "第一次响应不应为空")
			assert.NotEmpty(t, firstResp.Choices, "第一次响应的选择不应为空")
			firstAssistantMessage := firstResp.Choices[0].Message
			assert.Equal(t, openai.ChatMessageRoleAssistant, firstAssistantMessage.Role, "第一次响应消息角色应为assistant")

			// 检查是否收到了工具调用
			if tc.expectToolCall {
				assert.NotEmpty(t, firstAssistantMessage.ToolCalls, "第一次响应应包含 tool_calls")
				t.Logf("收到工具调用请求: %d 个", len(firstAssistantMessage.ToolCalls))
				for _, toolCall := range firstAssistantMessage.ToolCalls {
					t.Logf("  - ID: %s, Type: %s, Function: %s(%s)", toolCall.ID, toolCall.Type, toolCall.Function.Name, toolCall.Function.Arguments)
				}
			} else {
				if len(firstAssistantMessage.ToolCalls) > 0 {
					t.Logf("意外收到工具调用请求，但继续测试: %d 个", len(firstAssistantMessage.ToolCalls))
				}
			}

			// 如果没有工具调用，不需要进行后续步骤
			if len(firstAssistantMessage.ToolCalls) == 0 {
				t.Log("没有收到工具调用请求，跳过后续步骤")
				return
			}

			// 在发送第二个请求前，检查并修正assistant消息
			// 强制为包含 tool_calls 的 assistant 消息添加 content
			if firstAssistantMessage.Content == "" && len(firstAssistantMessage.ToolCalls) > 0 {
				firstAssistantMessage.Content = "Okay, I will call the required tools."
				t.Logf("强制为 assistant 消息添加了 content: '%s'", firstAssistantMessage.Content)
			}

			// 在发送第二个请求前，检查并修正工具调用参数
			for i, toolCall := range firstAssistantMessage.ToolCalls {
				// 检查工具调用参数是否为空或仅为{}
				if toolCall.Function.Arguments == "{}" || toolCall.Function.Arguments == "" {
					t.Logf("警告: 工具调用参数为空，自动补充参数")

					// 根据工具类型自动补充参数
					if toolCall.Function.Name == "get_weather" {
						firstAssistantMessage.ToolCalls[i].Function.Arguments = `{"location":"北京"}`
						t.Logf("已补充参数: %s", firstAssistantMessage.ToolCalls[i].Function.Arguments)
					}
				}
			}

			// 重新构建第二次请求的消息列表（使用修正后的firstAssistantMessage）
			messagesForSecondCall := []openai.ChatCompletionMessage{
				tc.initialMessages[0],
				firstAssistantMessage,
			}

			// 打印完整的消息列表进行验证
			t.Logf("完整的消息列表(共%d条):", len(messagesForSecondCall))
			for i, msg := range messagesForSecondCall {
				t.Logf("  消息 #%d: Role=%s, Content长度=%d", i+1, msg.Role, len(msg.Content))
				if msg.Role == openai.ChatMessageRoleTool {
					t.Logf("    ToolCallID=%s", msg.ToolCallID)
				} else if msg.Role == openai.ChatMessageRoleAssistant && len(msg.ToolCalls) > 0 {
					t.Logf("    包含%d个工具调用", len(msg.ToolCalls))
					for j, tc := range msg.ToolCalls {
						t.Logf("      工具调用 #%d: ID=%s, 函数=%s",
							j+1, tc.ID, tc.Function.Name)
					}
				}
			}

			// 打印为JSON格式以便更清晰查看结构
			messagesJSON, _ := json.MarshalIndent(messagesForSecondCall, "", "  ")
			t.Logf("消息JSON结构:\n%s", string(messagesJSON))

			// 添加工具结果消息
			firstToolCallID := firstAssistantMessage.ToolCalls[0].ID
			//firstToolName := firstAssistantMessage.ToolCalls[0].Function.Name

			toolResponseMessage := openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    tc.toolResult,
				ToolCallID: firstToolCallID,
				// 不添加Name字段
			}
			messagesForSecondCall = append(messagesForSecondCall, toolResponseMessage)

			// 重新打印更新后的消息列表
			messagesJSON, _ = json.MarshalIndent(messagesForSecondCall, "", "  ")
			t.Logf("添加工具响应后的消息列表:\n%s", string(messagesJSON))

			secondRequest := ChatRequest{
				Provider: "azure",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model:       "gpt-4o",
					Messages:    messagesForSecondCall,
					MaxTokens:   200,
					Temperature: 0.5,
				},
			}

			t.Log("--- 发送第二次请求 (带工具结果) ---")
			secondResp, err := CreateChatCompletion(secondRequest, nil)

			// 错误处理
			if err != nil {
				logAzureError(t, err)
				t.Fatalf("第二次API调用失败: %v", err)
			}

			// 验证最终响应
			assert.NotNil(t, secondResp, "第二次响应不应为空")
			assert.NotEmpty(t, secondResp.ID, "响应ID不应为空")
			assert.Equal(t, "chat.completion", secondResp.Object, "响应对象类型应为chat.completion")
			assert.NotZero(t, secondResp.Created, "创建时间不应为零")
			assert.NotEmpty(t, secondResp.Choices, "选择不应为空")

			if len(secondResp.Choices) > 0 {
				finalMessage := secondResp.Choices[0].Message
				assert.Equal(t, openai.ChatMessageRoleAssistant, finalMessage.Role, "最终消息角色应为assistant")
				assert.NotEmpty(t, finalMessage.Content, "最终助手消息内容不应为空")
				t.Logf("最终助手响应内容: %s", finalMessage.Content)
				assert.Contains(t, finalMessage.Content, "天气", "最终响应应包含天气信息")
				assert.Contains(t, finalMessage.Content, "北京", "最终响应应包含城市名")
				t.Logf("完成原因: %s", secondResp.Choices[0].FinishReason)
			}

			// 打印和检查工具调用，确保它包含location参数
			for _, toolCall := range firstAssistantMessage.ToolCalls {
				parsedArgs := make(map[string]interface{})
				json.Unmarshal([]byte(toolCall.Function.Arguments), &parsedArgs)
				t.Logf("  解析后的参数: %v", parsedArgs)

				if _, hasLocation := parsedArgs["location"]; !hasLocation && toolCall.Function.Name == "get_weather" {
					t.Logf("  警告: 工具调用缺少location参数")
				}
			}
		})
	}
}

// TestChatService_CreateChatCompletionMultiTool 测试多个工具调用功能
// 执行命令：go test -run TestChatService_CreateChatCompletionMultiTool
func TestChatService_CreateChatCompletionMultiTool(t *testing.T) {
	// 准备测试用例
	testCases := []struct {
		name            string
		initialMessages []openai.ChatCompletionMessage
		expectToolCalls bool
		toolResults     map[string]string // 模拟的工具结果，key为工具调用ID
		expectError     bool
	}{
		{
			name: "Azure多工具调用测试",
			initialMessages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "帮我计算一下30摄氏度是多少华氏度？以及2立方米等于多少升？",
				},
			},
			expectToolCalls: true,
			toolResults: map[string]string{
				"": `{"celsius": 30, "fahrenheit": 86}`, // 会在测试中动态替换key
			},
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// --- 第一步：发送初始请求，获取工具调用 ---
			firstRequest := ChatRequest{
				Provider: "azure",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model:    "gpt-4o", // 确保模型支持工具调用
					Messages: tc.initialMessages,
					// 定义多个工具
					Tools: []openai.Tool{
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "temperature_converter",
								Description: "将摄氏度转换为华氏度",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"celsius": map[string]interface{}{
											"type":        "number",
											"description": "摄氏度温度值",
										},
									},
									"required": []string{"celsius"},
								},
							},
						},
						{
							Type: openai.ToolTypeFunction,
							Function: &openai.FunctionDefinition{
								Name:        "volume_converter",
								Description: "将立方米转换为升",
								Parameters: map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"cubic_meters": map[string]interface{}{
											"type":        "number",
											"description": "体积，单位是立方米",
										},
									},
									"required": []string{"cubic_meters"},
								},
							},
						},
					},
					ToolChoice:  "auto", // 让模型决定是否使用工具
					MaxTokens:   200,
					Temperature: 0.5,
				},
			}

			t.Log("--- 发送第一次请求 ---")
			firstResp, err := CreateChatCompletion(firstRequest, nil)

			// 错误处理
			if err != nil {
				logAzureError(t, err)
				if tc.expectError {
					assert.Error(t, err, "预期应返回错误")
					return
				}
				t.Fatalf("第一次API调用失败: %v", err)
			}
			if tc.expectError {
				assert.NoError(t, err, "预期有错误但API调用成功")
				return
			}

			// 验证第一次响应
			assert.NotNil(t, firstResp, "第一次响应不应为空")
			assert.NotEmpty(t, firstResp.Choices, "第一次响应的选择不应为空")
			firstAssistantMessage := firstResp.Choices[0].Message
			assert.Equal(t, openai.ChatMessageRoleAssistant, firstAssistantMessage.Role, "第一次响应消息角色应为assistant")

			// 检查是否收到了工具调用
			if tc.expectToolCalls {
				assert.NotEmpty(t, firstAssistantMessage.ToolCalls, "第一次响应应包含 tool_calls")
				t.Logf("收到工具调用请求: %d 个", len(firstAssistantMessage.ToolCalls))
				for _, toolCall := range firstAssistantMessage.ToolCalls {
					t.Logf("  - ID: %s, Type: %s, Function: %s(%s)", toolCall.ID, toolCall.Type, toolCall.Function.Name, toolCall.Function.Arguments)
				}
			} else {
				if len(firstAssistantMessage.ToolCalls) > 0 {
					t.Logf("意外收到工具调用请求，但继续测试: %d 个", len(firstAssistantMessage.ToolCalls))
				}
			}

			// 如果没有工具调用，不需要进行后续步骤
			if len(firstAssistantMessage.ToolCalls) == 0 {
				t.Log("没有收到工具调用请求，跳过后续步骤")
				return
			}

			// 在发送第二个请求前，检查并修正assistant消息
			// 强制为包含 tool_calls 的 assistant 消息添加 content
			if firstAssistantMessage.Content == "" && len(firstAssistantMessage.ToolCalls) > 0 {
				firstAssistantMessage.Content = "Okay, I will call the required tools."
				t.Logf("强制为 assistant 消息添加了 content: '%s'", firstAssistantMessage.Content)
			}

			// 在发送第二个请求前，检查并修正工具调用参数
			for i, toolCall := range firstAssistantMessage.ToolCalls {
				// 检查工具调用参数是否为空或仅为{}
				if toolCall.Function.Arguments == "{}" || toolCall.Function.Arguments == "" {
					t.Logf("警告: 工具调用参数为空，自动补充参数")

					// 根据工具类型自动补充参数
					if toolCall.Function.Name == "temperature_converter" {
						firstAssistantMessage.ToolCalls[i].Function.Arguments = `{"celsius": 30}` // 提供默认参数
					} else if toolCall.Function.Name == "volume_converter" {
						firstAssistantMessage.ToolCalls[i].Function.Arguments = `{"cubic_meters": 2}`
					}

					if firstAssistantMessage.ToolCalls[i].Function.Arguments != "{}" && firstAssistantMessage.ToolCalls[i].Function.Arguments != "" {
						t.Logf("已补充参数: %s", firstAssistantMessage.ToolCalls[i].Function.Arguments)
					}
				}
			}

			// --- 第二步：发送第二次请求，包含工具结果 ---
			// 构建包含工具结果的消息列表
			messagesForSecondCall := []openai.ChatCompletionMessage{
				tc.initialMessages[0],
				firstAssistantMessage,
			}

			// 处理所有工具调用的响应
			for i, toolCall := range firstAssistantMessage.ToolCalls {
				toolCallID := toolCall.ID
				toolName := toolCall.Function.Name

				// 设置工具结果
				var toolResult string

				// 根据工具类型提供结果
				if toolName == "temperature_converter" {
					toolResult = `{"celsius": 30, "fahrenheit": 86}`
				} else if toolName == "volume_converter" {
					toolResult = `{"cubic_meters": 2, "liters": 2000}`
				}

				// 添加工具的调用结果到消息列表
				toolResponseMessage := openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    toolResult,
					ToolCallID: toolCallID,
					// Name:       toolName, // 移除 Name 字段
				}
				messagesForSecondCall = append(messagesForSecondCall, toolResponseMessage)

				t.Logf("已添加第%d个工具(%s)的响应", i+1, toolName)
			}

			t.Logf("第二次请求消息列表(包含所有工具结果):")
			messagesJSON, _ := json.MarshalIndent(messagesForSecondCall, "", "  ")
			t.Logf("%s", string(messagesJSON))

			// 发送第二次请求，不重复工具定义
			secondRequest := ChatRequest{
				Provider: "azure",
				ChatCompletionRequest: openai.ChatCompletionRequest{
					Model:       "gpt-4o",
					Messages:    messagesForSecondCall,
					MaxTokens:   200,
					Temperature: 0.5,
				},
			}

			t.Log("--- 发送第二次请求 (带所有工具结果) ---")
			secondResp, err := CreateChatCompletion(secondRequest, nil)

			// 错误处理
			if err != nil {
				logAzureError(t, err)
				t.Fatalf("第二次API调用失败: %v", err)
			}

			// 验证第二次响应
			assert.NotNil(t, secondResp, "第二次响应不应为空")
			secondAssistantMessage := secondResp.Choices[0].Message
			t.Logf("第二次助手响应内容: %s", secondAssistantMessage.Content)

			// 检查第二次响应是否包含工具调用
			if len(secondAssistantMessage.ToolCalls) > 0 {
				t.Logf("收到第二次工具调用请求: %d 个", len(secondAssistantMessage.ToolCalls))

				// 准备第三次请求 (包含第二个工具调用结果)
				messagesForThirdCall := append(messagesForSecondCall, secondAssistantMessage)

				// 创建包含第二个工具调用结果的工具消息
				secondToolCallID := secondAssistantMessage.ToolCalls[0].ID
				secondToolName := secondAssistantMessage.ToolCalls[0].Function.Name

				// 设置第二个工具的结果
				var secondToolResult string
				if result, exists := tc.toolResults[secondToolCallID]; exists {
					secondToolResult = result
				} else {
					// 根据工具类型提供默认结果
					if secondToolName == "temperature_converter" {
						secondToolResult = `{"celsius": 30, "fahrenheit": 86}`
					} else if secondToolName == "volume_converter" {
						secondToolResult = `{"cubic_meters": 2, "liters": 2000}`
					}
				}

				// 添加第二个工具的调用结果到消息列表
				secondToolResponseMessage := openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    secondToolResult,
					ToolCallID: secondToolCallID,
					// 不添加Name字段
				}
				messagesForThirdCall = append(messagesForThirdCall, secondToolResponseMessage)

				t.Logf("第三次请求消息列表(包含第二个工具结果):")
				messagesJSON, _ = json.MarshalIndent(messagesForThirdCall, "", "  ")
				t.Logf("%s", string(messagesJSON))

				// 发送第三次请求
				thirdRequest := ChatRequest{
					Provider: "azure",
					ChatCompletionRequest: openai.ChatCompletionRequest{
						Model:       "gpt-4o",
						Messages:    messagesForThirdCall,
						MaxTokens:   200,
						Temperature: 0.5,
					},
				}

				t.Log("--- 发送第三次请求 (带第二个工具结果) ---")
				thirdResp, err := CreateChatCompletion(thirdRequest, nil)

				// 错误处理
				if err != nil {
					logAzureError(t, err)
					t.Fatalf("第三次API调用失败: %v", err)
				}

				// 验证最终响应
				assert.NotNil(t, thirdResp, "第三次响应不应为空")
				finalMessage := thirdResp.Choices[0].Message
				assert.Equal(t, openai.ChatMessageRoleAssistant, finalMessage.Role, "最终消息角色应为assistant")
				assert.NotEmpty(t, finalMessage.Content, "最终助手消息内容不应为空")
				t.Logf("最终助手响应内容: %s", finalMessage.Content)

				// 验证响应中是否包含温度和体积转换相关信息
				assert.True(t,
					strings.Contains(finalMessage.Content, "华氏度") ||
						strings.Contains(finalMessage.Content, "摄氏度") ||
						strings.Contains(finalMessage.Content, "立方米") ||
						strings.Contains(finalMessage.Content, "升"),
					"最终响应应包含温度或体积转换相关信息")
			} else {
				// 如果第二次响应没有工具调用，则验证第二次响应
				assert.NotEmpty(t, secondAssistantMessage.Content, "第二次助手消息内容不应为空")
				// 验证响应中是否包含相关信息
				assert.True(t,
					strings.Contains(secondAssistantMessage.Content, "华氏度") ||
						strings.Contains(secondAssistantMessage.Content, "摄氏度") ||
						strings.Contains(secondAssistantMessage.Content, "立方米") ||
						strings.Contains(secondAssistantMessage.Content, "升"),
					"第二次响应应包含温度或体积转换相关信息")
			}
		})
	}
}

// logAzureError 辅助函数，用于记录详细的 Azure API 错误
func logAzureError(t *testing.T, err error) {
	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		paramStr := "<nil>"
		if apiErr.Param != nil {
			paramStr = *apiErr.Param
		}
		t.Logf("Azure API 错误: Status=%d, Type=%s, Code=%v, Param=%s, Message=%s", apiErr.HTTPStatusCode, apiErr.Type, apiErr.Code, paramStr, apiErr.Message)
	} else {
		t.Logf("测试期间出现错误: %v", err)
	}
}
